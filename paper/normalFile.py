# mcp_toolroute_zero_mem_exact_usage.py
"""
Multi-Server MCP Client (ZERO MEMORY, Exact Usage)
- Zero-memory per query
- Plain-English tool listing
- Token counting via OpenAI API Response (Source of Truth)
- Sanitized tool results
- Retry-on-error with sanitized messages
- Safe shutdown wrapper
"""

import json
import logging
import anyio
# tiktoken removed: we now rely on API response for counts
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Dict, Any, List, Optional, Tuple

from dotenv import load_dotenv
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from openai import OpenAI

# ---------------------------
# CONFIG: MCP SERVERS
# ---------------------------
KLAVIS_SERVERS = {
    "gmail": "https://gmail-mcp-server.klavis.ai/mcp/?instance_id=e0ddd5ee-45fc-4791-b9f7-5a04d8a58463",
    "github": "https://strata.klavis.ai/mcp/?instance_id=df9ad3af-9eb3-4287-b2f4-acbaa5db1138",
    "linear": "https://linear-mcp-server.klavis.ai/mcp/?instance_id=8e711cd1-909a-4641-95e7-b3d5ee358110",
    "gcalendar": "https://gcalendar-mcp-server.klavis.ai/mcp/?instance_id=9d9a4b34-d0c5-4b8e-b633-1aa101f57de6",
    "gdrive": "https://gdrive-mcp-server.klavis.ai/mcp/?instance_id=ab124495-a682-407e-b9a2-d82bb8ab77d0",
    "jira": "https://strata.klavis.ai/mcp/?instance_id=1c92c41b-f007-4c4b-81ab-be0a6be9b0aa",
    "notion": "https://strata.klavis.ai/mcp/?instance_id=35468960-6bec-4581-b141-dccd41e87742",
    "slack": "https://slack-mcp-server.klavis.ai/mcp/?instance_id=bfb2bca1-e73b-4c9d-9338-de1ecd35f4ea",
    "attio": "https://attio-mcp-server.klavis.ai/mcp/?instance_id=82c94b49-2981-4ab2-9433-1f081a36f22c",
    "hackerNews": "https://hacker-news-mcp-server.klavis.ai/mcp/?instance_id=000c2de5-7296-4417-97e2-6082e77a0050",
    "youtube": "https://youtube-mcp-server.klavis.ai/mcp/?instance_id=e5f0b026-f5db-402e-a6b6-62000d3444ab"
}

load_dotenv()

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("ZEROMEM-MCP")


# ---------------------------
# Safe cleanup wrapper
# ---------------------------
@asynccontextmanager
async def safe_stack():
    try:
        yield
    except Exception:
        pass


# ---------------------------
# Helpers
# ---------------------------
def safe_json_dump(obj: Any, max_len: int = 2000) -> str:
    try:
        s = json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        s = repr(obj)
    if len(s) > max_len:
        return s[:max_len] + "...(truncated)"
    return s


def sanitize_tool_result(result_obj: Any, max_chars: int = 8000) -> str:
    """Return readable plain text for tool results; truncate if needed."""
    if result_obj is None:
        return ""
    try:
        if hasattr(result_obj, "content") and isinstance(result_obj.content, (list, tuple)):
            parts = []
            for item in result_obj.content:
                if hasattr(item, "text"):
                    parts.append(str(item.text))
                elif isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(safe_json_dump(item, max_len=500))
            joined = "\n".join(parts)
            return joined[:max_chars] + ("...(truncated)" if len(joined) > max_chars else joined)
    except Exception:
        pass
    try:
        s = json.dumps(result_obj, default=str, ensure_ascii=False)
    except Exception:
        s = str(result_obj)
    if len(s) > max_chars:
        return s[:max_chars] + "...(truncated)"
    return s


def pretty_print_tools_plain(tools: List[Dict[str, Any]]):
    if not tools:
        log.info("No tools available.")
        return
    log.info(f"Tools available to AI: {len(tools)}")
    for idx, tf in enumerate(tools, start=1):
        func = tf.get("function", {})
        name = func.get("name", "<unknown>")
        desc = func.get("description", "")
        log.info(f"  {idx}. {name}")
        if desc:
            desc_line = "    Desc: " + (desc.replace("\n", " ")[:300] + ("..." if len(desc) > 300 else ""))
            log.info(desc_line)


# ---------------------------
# Client
# ---------------------------
class MultiMCPClient:
    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.openai = OpenAI()

    async def process_query(self, query: str):
        log.info("--------------------------------------------------")
        log.info(f"New query: '{query}'")

        # Reset per-query state
        self.messages: List[Dict[str, Any]] = []
        total_input_tokens = 0
        total_output_tokens = 0

        # Add user message
        user_msg = {"role": "user", "content": query}
        self.messages.append(user_msg)

        # Build tools list
        tools_payload: List[Dict[str, Any]] = []
        for name, session in self.sessions.items():
            try:
                resp = await session.list_tools()
                server_tools = getattr(resp, "tools", []) or []
                for t in server_tools:
                    tools_payload.append({
                        "type": "function",
                        "function": {
                            "name": f"{name}_{t.name}",
                            "description": getattr(t, "description", "") or "",
                            "parameters": getattr(t, "inputSchema", None)
                        }
                    })
            except Exception as e:
                log.warning(f"Could not list tools from {name}: {e}")

        pretty_print_tools_plain(tools_payload)

        # Helper: sanitized retry builder
        def build_sanitized_messages(msgs: List[Dict[str, Any]], content_limit: int = 2000) -> List[Dict[str, Any]]:
            sanitized = []
            for m in msgs:
                entry = {"role": m.get("role", "")}
                if "tool_call_id" in m:
                    entry["tool_call_id"] = m["tool_call_id"]
                content = str(m.get("content", "") or "")
                if len(content) > content_limit:
                    content = content[:content_limit] + "...(truncated)"
                entry["content"] = content
                # preserve minimal tool_calls
                if "tool_calls" in m and isinstance(m["tool_calls"], (list, tuple)):
                    tc_list = []
                    for tc in m["tool_calls"]:
                        try:
                            if isinstance(tc, dict):
                                func = tc.get("function", {})
                                tc_id = tc.get("id")
                                fname = func.get("name")
                                args = func.get("arguments")
                            else:
                                tc_id = getattr(tc, "id", None)
                                fname = getattr(getattr(tc, "function", None), "name", None)
                                args = getattr(getattr(tc, "function", None), "arguments", None)
                            args_str = str(args)[:200] if args is not None else ""
                            tc_list.append({
                                "id": tc_id,
                                "type": "function",
                                "function": {"name": fname, "arguments": args_str}
                            })
                        except Exception:
                            tc_list.append(
                                {"id": None, "type": "function", "function": {"name": None, "arguments": ""}})
                    entry["tool_calls"] = tc_list
                sanitized.append(entry)
            return sanitized

        # OpenAI call extracting EXACT usage from response
        def openai_call_with_retry(messages_payload: List[Dict[str, Any]], tools_payload_local: List[Dict[str, Any]],
                                   label: str = "initial") -> Tuple[Any, int, int]:
            log.info(f"Sending request to OpenAI ({label})...")

            try:
                resp = self.openai.chat.completions.create(
                    model="gpt-4.1-mini-2025-04-14",
                    messages=messages_payload,
                    tools=tools_payload_local
                )
                # Extract exact usage
                usage = getattr(resp, "usage", None)
                in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
                out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
                log.info(f"  -> Usage: {in_tok} input / {out_tok} output")
                return resp, in_tok, out_tok

            except Exception as e:
                log.warning(f"OpenAI call failed ({label}): {e}. Attempting sanitized retry.")
                sanitized = build_sanitized_messages(messages_payload, content_limit=2000)

                resp2 = self.openai.chat.completions.create(
                    model="gpt-4.1-mini-2025-04-14",
                    messages=sanitized,
                    tools=tools_payload_local
                )
                # Extract exact usage from retry
                usage2 = getattr(resp2, "usage", None)
                in_tok2 = getattr(usage2, "prompt_tokens", 0) if usage2 else 0
                out_tok2 = getattr(usage2, "completion_tokens", 0) if usage2 else 0
                log.info(f"  -> Retry Usage: {in_tok2} input / {out_tok2} output")
                return resp2, in_tok2, out_tok2

        # ---------- initial AI call ----------
        try:
            resp, in_tok, out_tok = openai_call_with_retry(self.messages, tools_payload, label="initial")
            total_input_tokens += in_tok
            total_output_tokens += out_tok
        except Exception as e:
            log.error(f"Initial OpenAI call failed permanently: {e}")
            return "OpenAI call failed."

        # extract assistant message
        try:
            assistant_msg = resp.choices[0].message
        except Exception:
            choices = getattr(resp, "choices", []) or []
            assistant_msg = choices[0].get("message") if choices else {"content": ""}

        assistant_content = getattr(assistant_msg, "content", None) or (
            assistant_msg.get("content") if isinstance(assistant_msg, dict) else "")

        # normalize tool_calls
        tool_calls = getattr(assistant_msg, "tool_calls", None)
        if tool_calls is None and isinstance(assistant_msg, dict):
            tool_calls = assistant_msg.get("tool_calls", [])
        tool_calls = tool_calls or []
        log.info(f"AI requested {len(tool_calls)} tool calls")

        # append assistant entry
        assistant_entry = {"role": "assistant", "content": assistant_content}
        if tool_calls:
            safe_calls = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    tc_copy = dict(tc)
                    if "type" not in tc_copy: tc_copy["type"] = "function"
                    safe_calls.append(tc_copy)
                else:
                    func = getattr(tc, "function", None)
                    tc_id = getattr(tc, "id", None)
                    fname = getattr(func, "name", None) if func else None
                    args = getattr(func, "arguments", None) if func else None
                    safe_calls.append({"id": tc_id, "type": "function", "function": {"name": fname, "arguments": args}})
            assistant_entry["tool_calls"] = safe_calls
        self.messages.append(assistant_entry)

        # ---------- Agentic loop ----------
        round_idx = 0
        while tool_calls:
            round_idx += 1
            log.info(f"--- Agentic loop round {round_idx} (processing {len(tool_calls)} calls) ---")

            for tc in tool_calls:
                # defensive parsing
                if isinstance(tc, dict):
                    func = tc.get("function", {})
                    full_name = func.get("name")
                    raw_args = func.get("arguments")
                    tc_id = tc.get("id")
                else:
                    func = getattr(tc, "function", None)
                    full_name = getattr(func, "name", None) if func else None
                    raw_args = getattr(func, "arguments", None) if func else None
                    tc_id = getattr(tc, "id", None)

                if not full_name or "_" not in full_name:
                    log.warning(f"Skipping malformed tool call name: {full_name}")
                    self.messages.append(
                        {"role": "tool", "tool_call_id": tc_id, "content": f"Error: malformed name {full_name}"})
                    continue

                service, tool_name = full_name.split("_", 1)
                session = self.sessions.get(service)
                if not session:
                    log.warning(f"Service '{service}' not connected.")
                    self.messages.append(
                        {"role": "tool", "tool_call_id": tc_id, "content": f"Error: service {service} not connected"})
                    continue

                # parse args
                args = {}
                if raw_args:
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        args = {"_raw": str(raw_args)}

                # execute
                log.info(f"Running tool: {service}.{tool_name}")
                try:
                    result = await session.call_tool(tool_name, args)
                    text = sanitize_tool_result(result, max_chars=8000)
                    log.info(f"Tool result length: {len(text)} chars")
                except Exception as e:
                    text = f"Tool execution error: {e}"
                    log.error(text)

                self.messages.append({"role": "tool", "tool_call_id": tc_id, "content": text})

            # follow-up call
            try:
                resp2, in_tok2, out_tok2 = openai_call_with_retry(self.messages, tools_payload,
                                                                  label=f"loop_{round_idx}")
                total_input_tokens += in_tok2
                total_output_tokens += out_tok2
            except Exception as e:
                log.error(f"OpenAI follow-up failed: {e}")
                break

            try:
                assistant_msg = resp2.choices[0].message
            except Exception:
                choices = getattr(resp2, "choices", []) or []
                assistant_msg = choices[0].get("message") if choices else {"content": ""}

            assistant_content = getattr(assistant_msg, "content", None) or (
                assistant_msg.get("content") if isinstance(assistant_msg, dict) else "")

            tool_calls = getattr(assistant_msg, "tool_calls", None)
            if tool_calls is None and isinstance(assistant_msg, dict):
                tool_calls = assistant_msg.get("tool_calls", [])
            tool_calls = tool_calls or []

            assistant_entry = {"role": "assistant", "content": assistant_content}
            if tool_calls:
                safe_calls = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        tc_copy = dict(tc)
                        if "type" not in tc_copy: tc_copy["type"] = "function"
                        safe_calls.append(tc_copy)
                    else:
                        func = getattr(tc, "function", None)
                        tc_id = getattr(tc, "id", None)
                        fname = getattr(func, "name", None) if func else None
                        args = getattr(func, "arguments", None) if func else None
                        safe_calls.append(
                            {"id": tc_id, "type": "function", "function": {"name": fname, "arguments": args}})
                assistant_entry["tool_calls"] = safe_calls
            self.messages.append(assistant_entry)
            log.info(f"AI requested {len(tool_calls)} tool calls in follow-up")

        # ---------- done ----------
        log.info("Agentic loop complete.")
        log.info(f"Total Input Tokens: {total_input_tokens}")
        log.info(f"Total Output Tokens: {total_output_tokens}")
        log.info(f"Total Combined: {total_input_tokens + total_output_tokens}")

        final_text = self.messages[-1].get("content") if self.messages else "(no response)"
        return final_text

    async def chat_loop(self):
        print("MCP Client Ready.\n")
        while True:
            q = await anyio.to_thread.run_sync(input, "\nQuery: ")
            q = q.strip()
            if q.lower() in ("quit", "exit"):
                break
            out = await self.process_query(q)
            print("\nAssistant:", out)


# ---------------------------
# Main
# ---------------------------
async def main():
    client = MultiMCPClient()
    async with safe_stack(), AsyncExitStack() as stack:
        for name, url in KLAVIS_SERVERS.items():
            try:
                streams = await stack.enter_async_context(streamablehttp_client(url))
                session = await stack.enter_async_context(ClientSession(streams[0], streams[1]))
                await session.initialize()
                client.sessions[name] = session
                log.info(f"Connected to {name}")
            except Exception as e:
                log.warning(f"Failed to connect to {name}: {e}")

        print("MCP Client Ready.\n")
        await client.chat_loop()


if __name__ == "__main__":
    anyio.run(main, backend="trio")