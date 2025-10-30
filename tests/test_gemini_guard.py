"""
MCP Gemini Client with GeminiGuard (using google.genai)

Usage:
    python gemini_mcp_client.py http://localhost:5000/mcp
"""

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import logging
from functools import partial
from typing import Optional
import anyio
from dotenv import load_dotenv
from google import genai
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from sk_guardrails.guards.geminiGuard import GeminiGuard
import os

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gemini_client")

# Load Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env")

client = genai.Client(api_key=GEMINI_API_KEY)


class MCPGeminiClient:
    def __init__(self, guard: GeminiGuard):
        self.session: Optional[ClientSession] = None
        self.history = []
        self.guard = guard

    async def get_tools_schema(self):
        response = await self.session.list_tools()
        tools = [{
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema
        } for tool in response.tools]
        return tools

    def format_prompt(self, query: str, tools: list) -> str:
        prompt = "You are an assistant. Available tools:\n"
        for tool in tools:
            prompt += f"- {tool['name']}: {tool['description']}\n"
        prompt += "\nBased on user's query, you can choose to call a tool.\n"
        prompt += f"User: {query}\n"
        prompt += "Reply with either your answer or a tool call like:\n"
        prompt += "`CALL <tool_name> <json_args>`\n"
        return prompt

    def parse_tool_call(self, response: str):
        if response.startswith("CALL"):
            try:
                _, name, args_str = response.strip().split(" ", 2)
                args = json.loads(args_str)
                return name, args
            except Exception:
                return None
        return None

    async def process_query(self, query: str) -> str:
        tools = await self.get_tools_schema()
        prompt = self.format_prompt(query, tools)

        # Use the GenAI client
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt]
        )
        result = response.text.strip()
        final_text = []

        tool_call = self.parse_tool_call(result)
        if tool_call:
            tool_name, tool_args = tool_call
            # Step 1: Check tool usage with GeminiGuard
            is_valid = await self.guard.check(tool_name, tool_args)
            if not is_valid:
                final_text.append(f"❌ Tool call rejected by GeminiGuard: {tool_name}({tool_args})")
            else:
                final_text.append(f"✅ Tool call approved by GeminiGuard: {tool_name}({tool_args})")
                # Step 2: Execute the tool if valid
                result_obj = await self.session.call_tool(tool_name, tool_args)
                tool_output = result_obj.content[0].text
                final_text.append(f"[Tool call result: {tool_output}]")
                # Step 3: Let Gemini summarize / continue the reply
                follow_up = client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=[
                        f"User asked: {query}",
                        f"Tool {tool_name} responded: {tool_output}",
                        "Now summarize or continue the reply:"
                    ]
                )
                final_text.append(follow_up.text.strip())
        else:
            # Normal AI response
            final_text.append(result)

        return "\n".join(final_text)

    async def chat_loop(self):
        print("\nMCP Gemini Client Started!")
        print("Type your queries, 'clear' to clear history, or 'quit' to exit.")

        while True:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break
            elif query.lower() == 'clear':
                self.history = []
                print("History cleared.")
                continue

            try:
                response = await self.process_query(query)
                print("\n" + response)
            except Exception as e:
                print(f"\nError: {str(e)}")


async def run_session(read_stream, write_stream):
    # Initialize GeminiGuard
    guard = GeminiGuard(
        mcp_server_url=os.getenv("MCP_URL", "http://localhost:5000/mcp"),
        gemini_model="gemini-2.5-flash",
        api_key=GEMINI_API_KEY
    )
    await guard.initialize()
    print(f"✅ GeminiGuard initialized. Loaded {len(guard.tool_specs)} tools from MCP.\n")

    client_instance = MCPGeminiClient(guard=guard)
    async with ClientSession(read_stream, write_stream) as session:
        client_instance.session = session
        logger.info("Initializing session")
        await session.initialize()
        logger.info("Initialized")
        await client_instance.chat_loop()


async def main(url: str):
    async with streamablehttp_client(url) as streams:
        await run_session(*streams[:2])


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="URL to connect to")
    args = parser.parse_args()
    anyio.run(partial(main, args.url), backend="trio")


if __name__ == "__main__":
    cli()
