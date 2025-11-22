# paper/guard.py

import logging
import json
import tiktoken
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

        try:
            self.encoding = tiktoken.encoding_for_model(openai_model)
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    # ----------------------------------------------------------
    # TOKEN HELPERS
    # ----------------------------------------------------------
    def count_text_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text or ""))

    def count_payload_tokens(self, messages) -> int:
        raw = json.dumps({
            "model": self.openai_model,
            "messages": messages
        })
        return len(self.encoding.encode(raw))

    # ----------------------------------------------------------
    # INITIALIZE
    # ----------------------------------------------------------
    async def initialize(self):
        """Fetch MCP metadata + init OpenAI client."""
        if not self.api_key:
            raise ValueError("OpenAI API key missing for Guard")

        logger.info("🔍 Fetching MCP tools for Guard...")
        self.tool_specs = await get_mcp_tools(self.mcp_server_url)
        logger.info(f"Loaded {len(self.tool_specs)} tools from MCP.")

        self.client = AsyncOpenAI(api_key=self.api_key)

    # ----------------------------------------------------------
    # VALIDATION (returns dict with token metrics)
    # ----------------------------------------------------------
    async def check_tool_usage(self, tool_name: str, input_data: Dict[str, Any]) -> dict:
        """
        Returns structured dict:
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
        if not tool_info:
            return {
                "verdict": "FAIL",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "raw_prompt_chars": 0,
                "raw_response_chars": 0
            }

        schema_str = json.dumps(tool_info["input_schema"], indent=2)

        # Build validation prompt
        prompt = f"""
        You are StreamKnight's validation AI. Your task is to determine if a proposed tool call is valid
        based on its definition.
        
        You are given the following information:
        ---
        Tool Name: {tool_name}
        Tool Description: {tool_info['description']}
        Tool Input Schema: {tool_info['input_schema']}
        Proposed Input: {input_data}
        ---
        
        Based on the tool's schema and description, is the proposed input valid and appropriate?
        The input must satisfy the schema's requirements (e.g., types, required fields).
        The values provided should make sense for the tool's intended purpose.
        
        Respond PASS or FAIL:
        'PASS' or 'FAIL'
"""

        raw_prompt_chars = len(prompt)

        messages = [{"role": "user", "content": prompt}]

        input_tokens = self.count_payload_tokens(messages)
        logger.info(f"[Guard] Input tokens ≈ {input_tokens}")

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
                "input_tokens": input_tokens,
                "output_tokens": 0,
                "total_tokens": input_tokens,
                "raw_prompt_chars": raw_prompt_chars,
                "raw_response_chars": 0
            }

        text = resp.choices[0].message.content.strip()
        raw_response_chars = len(text)

        output_tokens = self.count_text_tokens(text)
        total_tokens = input_tokens + output_tokens

        logger.info(f"[Guard] Output tokens ≈ {output_tokens}")
        logger.info(f"[Guard] Total tokens ≈ {total_tokens}")
        logger.info(f"[Guard] Raw response: {text}")

        # Normalize
        verdict = text.upper().replace('"', '').replace("'", "").strip()

        if verdict.startswith("PASS"):
            verdict = "PASS"
        elif verdict.startswith("FAIL"):
            verdict = "FAIL"
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
