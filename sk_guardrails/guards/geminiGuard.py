import asyncio
import logging
from typing import Dict, Any
from sk_guardrails.guards.utils.tool_inspector import get_mcp_tools
from google import genai

logger = logging.getLogger("gemini_guard")

class GeminiGuard:
    def __init__(self, mcp_server_url: str, gemini_model: str = "gemini-2.5-flash", api_key: str = None):
        self.mcp_server_url = mcp_server_url
        self.gemini_model = gemini_model
        self.api_key = api_key
        self.tool_specs = {}
        self.client = None

    async def initialize(self):
        """
        Load tool metadata from the MCP server and initialize Gemini client.
        """
        self.tool_specs = await get_mcp_tools(self.mcp_server_url)
        logger.info(f"Loaded {len(self.tool_specs)} tools from MCP server.")

        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            raise ValueError("Gemini API key not provided")

    async def verify_tool_call(self, tool_name: str, input_data: Dict[str, Any], output_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ask Gemini if the tool was used correctly.
        """
        tool_info = next((t for t in self.tool_specs if t["name"] == tool_name), None)
        if not tool_info:
            return {"verdict": "fail", "reason": f"Unknown tool: {tool_name}"}

        prompt = f"""
You are StreamKnight's verification AI.
You are given the following information:
---
Tool Name: {tool_name}
Tool Description: {tool_info['description']}
Tool Input Schema: {tool_info['input_schema']}
Actual Input: {input_data}
Actual Output: {output_data}
---

Determine if:
1. The input matches the tool's expected schema and purpose.
2. The output logically aligns with what the tool claims to do.
3. The tool appears to be used appropriately for the given intent.

Respond in JSON with:
{{"verdict": "pass|warn|fail", "reason": "<short reasoning>"}}
"""

        response = await self.client.aio.models.generate_content(
            model=self.gemini_model,
            contents=prompt
        )

        try:
            result = eval(response.text)  # expecting JSON
        except Exception:
            result = {"verdict": "warn", "reason": "Invalid model response"}

        logger.info(f"Gemini verdict for {tool_name}: {result}")
        return result

    async def monitor_event(self, event):
        """
        Accept a structured event like:
        {
          "tool": "create_meeting",
          "input": {...},
          "output": {...}
        }
        """
        result = await self.verify_tool_call(event["tool"], event["input"], event["output"])
        if result["verdict"] == "fail":
            logger.warning(f"❌ Tool misuse detected: {result['reason']}")
        elif result["verdict"] == "warn":
            logger.warning(f"⚠️ Potential issue: {result['reason']}")
        else:
            logger.info(f"✅ Tool call verified successfully: {event['tool']}")
