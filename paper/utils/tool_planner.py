import logging
import json
from openai import AsyncOpenAI

logger = logging.getLogger("tool_planner")


class ToolPlanner:
    def __init__(self, openai_model="gpt-4.1-mini", api_key=None):
        self.openai_model = openai_model
        self.api_key = api_key
        self.client = None

    async def initialize(self):
        if not self.api_key:
            raise ValueError("OpenAI API key missing")
        self.client = AsyncOpenAI(api_key=self.api_key)

    # ---------------------------
    # Build Tool Route (returns dict with metrics)
    # ---------------------------
    async def build_route(self, user_query: str, tool_specs: list) -> dict:
        """
        Returns:
            {
                "route": [...],
                "input_tokens": int,
                "output_tokens": int,
                "total_tokens": int,
                "raw_prompt_chars": int,
                "raw_response_chars": int
            }
        """

        # Build readable tool list
        tool_list_text = "\n".join(
            f"- {t['name']}: {t.get('description', '')}"
            for t in tool_specs
        )

        # Final planning prompt
        prompt = f"""
You are StreamKnight's planning AI.

Your job is to produce a correct tool execution route for the user's request.

User Query:
---
{user_query}
---

Available Tools:
{tool_list_text}

Rules:
1. Return ONLY a Python list of tool names, in correct execution order.
2. If a tool is used multiple times, return it every time it is needed.
    ex if sum is used 2 times in a row, return ["sum", "sum"]
2. No explanation.
3. No text outside the list.
4. Only use available tools.
5. If nothing needed, return [].

Example:
["tool", "tool"]
"""

        # Build messages for OpenAI
        messages = [{"role": "user", "content": prompt}]

        # Raw chars for metadata/debugging (cheap len() check)
        raw_prompt_chars = len(prompt)

        # ---- OpenAI call ----
        try:
            resp = await self.client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=0
            )
        except Exception as e:
            logger.error(f"❌ OpenAI planner error: {e}")
            return {
                "route": [],
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "raw_prompt_chars": raw_prompt_chars,
                "raw_response_chars": 0
            }

        # Extract EXACT usage from API response
        usage = getattr(resp, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        total_tokens = input_tokens + output_tokens

        logger.info(f"🟦 Planner usage: {input_tokens} in / {output_tokens} out")

        # Extract output text
        try:
            text = resp.choices[0].message.content.strip()
        except Exception:
            logger.error("❌ Planner returned no content")
            text = ""

        raw_response_chars = len(text)

        # Parse route safely
        try:
            # Ideally use json.loads, but keeping eval per your original logic
            # to handle Python-style single quotes if the model outputs them.
            route = eval(text)
            if not isinstance(route, list):
                raise ValueError("not a list")
        except Exception:
            logger.error("❌ Planner returned invalid list format")
            route = []

        # Final return dict
        return {
            "route": route,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "raw_prompt_chars": raw_prompt_chars,
            "raw_response_chars": raw_response_chars
        }