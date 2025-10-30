# guards/geminiGuard.py

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

    async def check_tool_usage(self, tool_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ask Gemini if the proposed tool call is valid.
        """
        tool_info = next((t for t in self.tool_specs if t["name"] == tool_name), None)
        if not tool_info:
            return {"verdict": "fail", "reason": f"Unknown tool: {tool_name}"}

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

        response = await self.client.aio.models.generate_content(
            model=self.gemini_model,
            contents=prompt
        )
        print(response.text)
        '''
        try:
            # Use a safer method to parse JSON
            import json
            result = json.loads(response.text)
        except (json.JSONDecodeError, TypeError):
            result = {"verdict": "fail", "reason": "Invalid model response format"}

        logger.info(f"Gemini check for {tool_name}: {result}")
        '''

        return response.text


    async def check(self, tool_name: str, input_data: Dict[str, Any]) -> bool:
        """
        Check if a tool call is valid. Returns True if the verdict is 'pass'.
        """
        result = await self.check_tool_usage(tool_name, input_data)
        if result == "PASS":
            logger.info(f"✅ Gemini approved tool call: {tool_name}")
            return True
        else:
            #reason = result.get("reason", "No reason provided")
            #logger.warning(f"❌ Gemini rejected tool call: {tool_name}. Reason: {reason}")
            return False
