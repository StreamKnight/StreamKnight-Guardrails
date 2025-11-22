# paper/guard.py

import logging
import json
# tiktoken removed: relying on API source of truth
from typing import Dict, Any
from paper.utils.tool_inspector import get_mcp_tools
from openai import AsyncOpenAI

logger = logging.getLogger("openai_guard")


class OpenAIGuard:
    def __init__(self, mcp_server_url: str,
                 openai_model: str = "gpt-4.1-mini",
                 api_key: str = None):

        self.mcp_server_url = mcp_server_url
        self.openai_model = openai_model
        self.api_key = api_key

        self.client: AsyncOpenAI | None = None
        self.tool_specs: list = []

    # ----------------------------------------------------------
    # INITIALIZE
    # ----------------------------------------------------------
    async def initialize(self):
        """Fetch MCP metadata + init OpenAI client."""
        if not self.api_key:
            raise ValueError("OpenAI API key missing for Guard")

        logger.info("🔍 Fetching MCP tools for Guard...")
        # Assuming get_mcp_tools handles connection safely;
        # if this requires the client session context, ensure it's passed correctly.
        # For this snippet, we assume it works as imported.
        try:
            self.tool_specs = await get_mcp_tools(self.mcp_server_url)
            logger.info(f"Loaded {len(self.tool_specs)} tools from MCP.")
        except Exception as e:
            logger.warning(f"Could not fetch MCP tools during init: {e}")

        self.client = AsyncOpenAI(api_key=self.api_key)

    # ----------------------------------------------------------
    # VALIDATION (returns dict with EXACT token metrics)
    # ----------------------------------------------------------
    async def check_tool_usage(self, tool_name: str, input_data: Dict[str, Any]) -> dict:
        """
        Returns structured dict with exact usage from API:
        {
            "verdict": "PASS"/"FAIL",
            "input_tokens": int,
            "output_tokens": int,
            "total_tokens": int,
            "raw_prompt_chars": int,
            "raw_response_chars": int
        }
        """

        tool_info = next((t for t in self.tool_specs if t["name"] == tool_name), None)

        # If tool definition not found, we can't validate schema, so FAIL safe.
        if not tool_info:
            # If we have no info, we might check if it's because tool_specs is empty
            # But generally safe to fail or pass depending on policy.
            # Defaulting to FAIL as per original logic.
            return {
                "verdict": "FAIL",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "raw_prompt_chars": 0,
                "raw_response_chars": 0
            }

        # Build validation prompt
        prompt = f"""
        You are StreamKnight's validation AI. Your task is to determine if a proposed tool call is valid
        based on its definition.

        You are given the following information:
        ---
        Tool Name: {tool_name}
        Tool Description: {tool_info.get('description', '')}
        Tool Input Schema: {tool_info.get('input_schema', {})}
        Proposed Input: {input_data}
        ---

        Based on the tool's schema and description, is the proposed input valid and appropriate?
        The input must satisfy the schema's requirements (e.g., types, required fields).
        The values provided should make sense for the tool's intended purpose.

        Respond PASS or FAIL:
        'PASS' or 'FAIL'
"""

        messages = [{"role": "user", "content": prompt}]

        # Raw char count (cheap metric, not tokens)
        raw_prompt_chars = len(prompt)

        # Call OpenAI
        try:
            resp = await self.client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=0
            )
        except Exception as e:
            logger.error(f"[Guard] OpenAI error: {e}")
            return {
                "verdict": "FAIL",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "raw_prompt_chars": raw_prompt_chars,
                "raw_response_chars": 0
            }

        # EXTRACT EXACT USAGE
        usage = getattr(resp, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        total_tokens = input_tokens + output_tokens

        # Process content
        text = resp.choices[0].message.content.strip()
        raw_response_chars = len(text)

        logger.info(f"[Guard] Usage: {input_tokens} in / {output_tokens} out")
        logger.info(f"[Guard] Raw response: {text}")

        # Normalize
        verdict_clean = text.upper().replace('"', '').replace("'", "").strip()
        if verdict_clean.startswith("PASS"):
            verdict = "PASS"
        else:
            verdict = "FAIL"

        return {
            "verdict": verdict,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "raw_prompt_chars": raw_prompt_chars,
            "raw_response_chars": raw_response_chars
        }

    # ----------------------------------------------------------
    # MAIN PUBLIC METHOD
    # ----------------------------------------------------------
    async def check(self, tool_name: str, input_data: Dict[str, Any]) -> bool:
        """Returns True/False but still logs internally."""
        result = await self.check_tool_usage(tool_name, input_data)

        if result["verdict"] == "PASS":
            logger.info(f"✅ Guard APPROVED: {tool_name}")
            return True

        logger.warning(f"❌ Guard REJECTED: {tool_name}")
        return False