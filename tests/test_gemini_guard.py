# tests/test_gemini_guard_real.py

import asyncio
import os
from sk_guardrails.guards.geminiGuard import GeminiGuard
from dotenv import load_dotenv

load_dotenv()

MCP_URL = os.getenv("MCP_TEST_URL", "http://localhost:5000/mcp")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

async def test_gemini_guard_real():
    # Initialize GeminiGuard
    guard = GeminiGuard(mcp_server_url=MCP_URL, gemini_model="gemini-2.5-flash", api_key=GEMINI_API_KEY)
    await guard.initialize()
    print(f"✅ GeminiGuard initialized. Loaded {len(guard.tool_specs)} tools from MCP.")

    if not guard.tool_specs:
        print("⚠️ No tools found on MCP server. Please check MCP URL.")
        return

    # Pick the first real tool from MCP server
    real_tool = guard.tool_specs[0]
    tool_name = real_tool["name"]
    description = real_tool["description"]
    input_schema = real_tool["input_schema"]

    print(f"Using tool for test: {tool_name}")
    print(f"Description: {description}")
    print(f"Input Schema: {input_schema}")

    # Define some test inputs
    valid_input = {}  # Replace with valid input based on the tool's schema
    invalid_input = {"wrong_field": "abc"}  # Intentionally invalid

    test_cases = [
        {"input_data": valid_input, "expected": True, "label": "Valid Input"},
        {"input_data": invalid_input, "expected": False, "label": "Invalid Input"}
    ]

    for case in test_cases:
        result = await guard.check(tool_name, case["input_data"])
        print(f"{case['label']} -> Verdict: {result}")

if __name__ == "__main__":
    asyncio.run(test_gemini_guard_real())
