"""
MCP Client with GeminiGuard integration.
This client connects to an MCP server, interacts with tools via OpenAI, and monitors
tool calls in real-time using GeminiGuard.
python test_geminiGuard.py http://localhost:5000/mcp
"""

import argparse
import json
import logging
from contextlib import AsyncExitStack
from functools import partial
from typing import Optional
import anyio
import sys
import os
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from dotenv import load_dotenv
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from openai import OpenAI
from sk_guardrails.guards.geminiGuard import GeminiGuard
from google import genai  # Gemini SDK

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MCP_URL = os.getenv("MCP_URL")

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("client")


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai = OpenAI()
        self.messages = []
        self.guard: Optional[GeminiGuard] = None

    async def initialize_guard(self):
        """Initialize GeminiGuard to monitor tool calls"""
        self.guard = GeminiGuard(MCP_URL, api_key=GEMINI_API_KEY)
        await self.guard.initialize()
        logger.info("GeminiGuard initialized and ready to monitor tools.")

    async def process_query(self, query: str) -> str:
        """Process a query using OpenAI and available tools"""
        self.messages.append({"role": "user", "content": query})

        # Get available tools from MCP server
        response = await self.session.list_tools()
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]

        # Call OpenAI chat with conversation history and tools
        response = self.openai.chat.completions.create(
            model="gpt-4.1-mini-2025-04-14",
            max_tokens=1000,
            messages=self.messages,
            tools=available_tools
        )

        final_text = []
        message = response.choices[0].message
        if message.content:
            final_text.append(message.content)

        # Process any tool calls
        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                # Execute tool
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                # Monitor tool call with GeminiGuard
                if self.guard:
                    event = {
                        "tool": tool_name,
                        "input": tool_args,
                        "output": str(result.content[0].text)  # or parse JSON if structured
                    }
                    await self.guard.monitor_event(event)

                # Add assistant message and tool result to history
                self.messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [{
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tool_call.function.arguments
                        }
                    }]
                })
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result.content[0].text)
                })
                final_text.append(f"[Tool call result: {result.content[0].text}]")

                # Continue conversation after tool call
                response = self.openai.chat.completions.create(
                    model="gpt-4.1-mini-2025-04-14",
                    max_tokens=1000,
                    messages=self.messages,
                )
                final_text.append(response.choices[0].message.content)

        if response.choices[0].message.content:
            self.messages.append({
                "role": "assistant",
                "content": response.choices[0].message.content
            })

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries, 'clear' to clear history, or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break
                elif query.lower() == "clear":
                    self.messages = []
                    print("Message history cleared.")
                    continue

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        await self.exit_stack.aclose()


async def run_session(read_stream, write_stream):
    client = MCPClient()
    async with ClientSession(read_stream, write_stream) as session:
        client.session = session
        logger.info("Initializing MCP session...")
        await session.initialize()
        logger.info("MCP session initialized.")

        # Initialize GeminiGuard
        await client.initialize_guard()

        # Start interactive chat loop
        await client.chat_loop()


async def main(url: str, args: list[str]):
    async with streamablehttp_client(url) as streams:
        await run_session(*streams[:2])


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="URL to connect to")
    parser.add_argument("args", nargs="*", help="Additional arguments")
    args = parser.parse_args()
    anyio.run(partial(main, args.url, args.args), backend="trio")


if __name__ == "__main__":
    cli()
