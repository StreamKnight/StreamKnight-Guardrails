# paper/guard.py

import logging
import json
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
        try:
            self.tool_specs = await get_mcp_tools(self.mcp_server_url)
            logger.info(f"Loaded {len(self.tool_specs)} tools from MCP.")
        except Exception as e:
            logger.warning(f"Could not fetch MCP tools during init: {e}")

        self.client = AsyncOpenAI(api_key=self.api_key)

    # ----------------------------------------------------------
    # VALIDATION (returns dict with EXACT token metrics + REASON)
    # ----------------------------------------------------------
    async def check_tool_usage(self, tool_name: str, input_data: Dict[str, Any]) -> dict:
        """
        Returns structured dict with exact usage and FAILURE REASON:
        {
            "verdict": "PASS"/"FAIL",
            "reason": "Explanation...",   <-- ADDED THIS
            "input_tokens": int,
            "output_tokens": int,
            "total_tokens": int,
            "raw": str                    <-- Raw response for debugging
        }
        """

        tool_info = next((t for t in self.tool_specs if t["name"] == tool_name), None)

        # If tool definition not found, FAIL with specific reason.
        if not tool_info:
            return {
                "verdict": "FAIL",
                "reason": f"Tool definition for '{tool_name}' not found in Guard specs.",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "raw": "Tool not found"
            }

        # --- UPDATED PROMPT ---
        # We explicitly ask for "FAIL: <Reason>"
        prompt = f"""
        You are StreamKnight's security and schema validator. 
        Analyze the proposed tool call against its definition.

        ---
        Tool Name: {tool_name}
        Tool Description: {tool_info.get('description', '')}
        Tool Input Schema: {tool_info.get('input_schema', {})}
        Proposed Input: {input_data}
        ---

        INSTRUCTIONS:
        1. check if the input satisfies the strict schema types and requirements.
        2. Check if the values are safe and appropriate for the tool's purpose.

        RESPONSE FORMAT:
        If VALID: Respond with exactly "PASS"
        If INVALID: Respond with "FAIL: <Short explanation of why it failed>"
        """

        messages = [{"role": "user", "content": prompt}]

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
                "reason": f"OpenAI API Error: {str(e)}",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "raw": str(e)
            }

        # EXTRACT USAGE
        usage = getattr(resp, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        total_tokens = input_tokens + output_tokens

        # --- PARSE REASONING ---
        text = resp.choices[0].message.content.strip()

        verdict = "FAIL"
        reason = text  # Default reason is the whole text

        # Check strictly for PASS/FAIL prefix
        if text.upper().startswith("PASS"):
            verdict = "PASS"
            reason = "Approved"
        elif text.upper().startswith("FAIL"):
            verdict = "FAIL"
            # Extract text after "FAIL:"
            parts = text.split(":", 1)
            if len(parts) > 1:
                reason = parts[1].strip()
            else:
                reason = text  # Fallback if they forgot the colon

        logger.info(f"[Guard] Verdict: {verdict} | Reason: {reason}")

        return {
            "verdict": verdict,
            "reason": reason,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "raw": text
        }

    # ----------------------------------------------------------
    # MAIN PUBLIC METHOD
    # ----------------------------------------------------------
    async def check(self, tool_name: str, input_data: Dict[str, Any]) -> bool:
        """Returns True/False but logs the REASON on failure."""
        result = await self.check_tool_usage(tool_name, input_data)

        if result["verdict"] == "PASS":
            logger.info(f"✅ Guard APPROVED: {tool_name}")
            return True

        # LOG THE REASON HERE
        logger.warning(f"❌ Guard REJECTED: {tool_name} -> REASON: {result['reason']}")
        return False