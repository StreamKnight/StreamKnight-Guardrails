import anyio
import sys
import os
import argparse
from functools import partial

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from google import genai
from dotenv import load_dotenv

load_dotenv()
# Load environment variables
MCP_URL = os.getenv("MCP_URL", "http://localhost:8000/mcp")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    sys.exit("GEMINI_API_KEY environment variable not set.")

# Create a GenAI client
client = genai.Client(api_key=GEMINI_API_KEY)

async def main(url: str):
    """
    Connects to an MCP server, retrieves a list of tools, and then enters a loop
    to send prompts to the Gemini API along with the available tools.
    """
    print("Connecting to MCP server at:", url)
    try:
        async with streamablehttp_client(url) as streams:
            async with ClientSession(*streams[:2]) as session:
                await session.initialize()
                print("MCP session initialized.")

                tools = await session.list_tools()
                print(f"Found {len(tools.tools)} tools.")

                gemini_tools = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    }
                    for tool in tools.tools
                ]

                while True:
                    # Run blocking input() in a separate thread
                    prompt = await anyio.to_thread.run_sync(input, "Enter prompt: ")
                    if prompt.lower() in ["quit", "exit"]:
                        break

                    # Run blocking Gemini API call in a separate thread
                    response = await anyio.to_thread.run_sync(
                        lambda: client.models.generate_content(
                            model="gemini-1.5-flash",
                            contents=[prompt]
                        )
                    )

                    print(response.text)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url", nargs="?", default=MCP_URL, help="URL of the MCP server")
    args = parser.parse_args()

    try:
        anyio.run(main, args.url)
    except KeyboardInterrupt:
        print("\nExiting...")
