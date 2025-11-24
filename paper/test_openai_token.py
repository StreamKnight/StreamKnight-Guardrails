# paper/test_openai_token.py
"""
Multi-Server MCP Client (ZERO MEMORY, Planner+Guard Controlled, STRICT MODE)
- Zero-memory per query
- Planner builds a route and the client exposes only the allowed tool each step
- Guard validates each tool call before execution
- STRICT MODE enabled for OpenAI structured outputs
- Token counting via OpenAI API Response (Source of Truth)
- Sanitized tool results
- Safe shutdown wrapper
"""
import os
import json
import logging
import anyio
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Dict, Any, List, Optional, Tuple

from dotenv import load_dotenv
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Defensive imports for user's planner & guard modules
try:
    from paper.utils.tool_planner import ToolPlanner as PaperToolPlanner
except Exception:
    PaperToolPlanner = None

try:
    from paper.guard import OpenAIGuard as PaperOpenAIGuard
except Exception:
    PaperOpenAIGuard = None

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

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("ZEROMEM-MCP-CTRL-STRICT")


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
    if result_obj is None: return ""
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


def sanitize_schema_for_strict(schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Recursively modifies a JSON schema to meet OpenAI's 'Strict Mode' requirements.
    """
    if schema is None:
        return {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

    if "type" not in schema and "properties" in schema:
        schema["type"] = "object"

    if schema.get("type") == "object":
        new_schema = schema.copy()
        new_schema["additionalProperties"] = False

        props = new_schema.get("properties", {})
        existing_required = new_schema.get("required", [])
        if isinstance(existing_required, list):
            all_keys = list(props.keys())
            new_schema["required"] = sorted(list(set(existing_required + all_keys)))

        new_props = {}
        for k, v in props.items():
            new_props[k] = sanitize_schema_for_strict(v)
        new_schema["properties"] = new_props
        return new_schema

    elif schema.get("type") == "array":
        new_schema = schema.copy()
        items = new_schema.get("items", {})
        new_schema["items"] = sanitize_schema_for_strict(items)
        return new_schema

    return schema


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
# Utilities
# ---------------------------
async def get_mcp_tools_from_sessions(sessions: Dict[str, ClientSession]) -> List[Dict[str, Any]]:
    """
    Fetch tools and convert them to OpenAI STRICT format.
    We maintain the 'Fat Object' (root level keys) for Planner/Guard compatibility,
    but ensure the inner 'function' dict is strictly valid for OpenAI.
    """
    all_tools = []
    for srv_name, session in sessions.items():
        try:
            resp = await session.list_tools()
            server_tools = getattr(resp, "tools", []) or []
            for t in server_tools:
                func_name = f"{srv_name}_{t.name}"

                raw_schema = getattr(t, "inputSchema", None)
                strict_schema = sanitize_schema_for_strict(raw_schema)

                entry = {
                    # OpenAI Strict Object
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "description": getattr(t, "description", "") or "",
                        "parameters": strict_schema,
                        "strict": True  # <--- Enabled Strict Mode
                    },
                    # Internal Compatibility Keys (for Planner/Guard adapters)
                    "name": func_name,
                    "description": getattr(t, "description", "") or "",
                    "input_schema": raw_schema  # Keep raw for internal logic if needed
                }
                all_tools.append(entry)
        except Exception:
            log.exception(f"Could not list tools from {srv_name} (ignored).")
    return all_tools


# ---------------------------
# Planner + Guard adapters
# ---------------------------
class PlannerAdapter:
    def __init__(self, openai_client: OpenAI, model: str = None, api_key: Optional[str] = None):
        self.openai = openai_client
        self.model = model or "gpt-4.1-mini-2025-04-14"
        self.api_key = api_key or OPENAI_API_KEY
        self._impl = None
        if PaperToolPlanner is not None:
            try:
                try:
                    self._impl = PaperToolPlanner(openai_model=self.model, api_key=self.api_key)
                except TypeError:
                    try:
                        self._impl = PaperToolPlanner(self.openai, model=self.model)
                    except Exception:
                        self._impl = PaperToolPlanner(api_key=self.api_key)
            except Exception:
                log.exception("Planner instantiation failed; planner disabled.")
                self._impl = None

    async def initialize(self):
        if self._impl is None: return
        init = getattr(self._impl, "initialize", None)
        if init:
            if asyncio.iscoroutinefunction(init):
                await init()
            else:
                try:
                    init()
                except Exception:
                    pass

    async def build_route(self, user_query: str, tool_specs: List[Dict[str, Any]]):
        if self._impl is None: return [], 0, 0
        build = getattr(self._impl, "build_route", None)
        if build is None: return [], 0, 0

        if asyncio.iscoroutinefunction(build):
            res = await build(user_query, tool_specs)
        else:
            res = build(user_query, tool_specs)

        if isinstance(res, dict):
            route = res.get("route", []) or res.get("routes", []) or []
            in_toks = int(res.get("input_tokens", 0) or 0)
            out_toks = int(res.get("output_tokens", 0) or 0)
            return route, in_toks, out_toks
        if isinstance(res, tuple):
            try:
                route = res[0] or []
                in_toks = int(res[1]) if len(res) > 1 else 0
                out_toks = int(res[2]) if len(res) > 2 else 0
                return route, in_toks, out_toks
            except Exception:
                return [], 0, 0
        if isinstance(res, list):
            return res, 0, 0
        return [], 0, 0


class GuardAdapter:
    def __init__(self, guard_impl=None, api_key: Optional[str] = None):
        self._impl = guard_impl
        self.api_key = api_key or OPENAI_API_KEY

    async def initialize(self, sessions: Dict[str, ClientSession]):
        if self._impl is None: return
        init = getattr(self._impl, "initialize", None)
        if init:
            if asyncio.iscoroutinefunction(init):
                await init()
            else:
                try:
                    init()
                except Exception:
                    pass

    async def check(self, tool_name: str, input_data: Dict[str, Any]):
        if self._impl is None:
            return True, 0, 0, "PASS (no guard)"

        ct = getattr(self._impl, "check_tool_usage", None)
        if ct:
            if asyncio.iscoroutinefunction(ct):
                res = await ct(tool_name, input_data)
            else:
                res = ct(tool_name, input_data)

            if isinstance(res, dict):
                verdict = res.get("verdict", "FAIL")
                in_toks = int(res.get("input_tokens", 0) or 0)
                out_toks = int(res.get("output_tokens", 0) or 0)
                raw = res.get("raw", res.get("raw_response", "") or res.get("raw_response_chars", "") or "")
                return (str(verdict).upper().startswith("PASS"), in_toks, out_toks, str(raw))
            txt = str(res or "")
            return (txt.upper().strip().startswith("PASS"), 0, 0, txt)

        c = getattr(self._impl, "check", None)
        if c:
            if asyncio.iscoroutinefunction(c):
                ok = await c(tool_name, input_data)
            else:
                ok = c(tool_name, input_data)
            return bool(ok), 0, 0, "PASS" if ok else "FAIL (bool)"
        return True, 0, 0, "PASS (no check method)"


# ---------------------------
# MCP Client
# ---------------------------
class MultiMCPClient:
    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.openai = OpenAI()
        self.planner_adapter = PlannerAdapter(self.openai, model=None)
        self.guard_adapter = None

    def build_sanitized_messages(self, msgs: List[Dict[str, Any]], content_limit: int = 2000) -> List[Dict[str, Any]]:
        sanitized = []
        for m in msgs:
            entry = {"role": m.get("role", "")}
            if "tool_call_id" in m:
                entry["tool_call_id"] = m["tool_call_id"]
            content = str(m.get("content", "") or "")
            if len(content) > content_limit:
                content = content[:content_limit] + "...(truncated)"
            entry["content"] = content
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
                        tc_list.append(
                            {"id": tc_id, "type": "function", "function": {"name": fname, "arguments": args_str}})
                    except Exception:
                        tc_list.append({"id": None, "type": "function", "function": {"name": None, "arguments": ""}})
                entry["tool_calls"] = tc_list
            sanitized.append(entry)
        return sanitized

    def openai_call_with_retry(self, messages_payload: List[Dict[str, Any]], tools_payload: List[Dict[str, Any]],
                               label: str = "initial") -> Tuple[Any, int, int]:
        log.info(f"{label.capitalize()} call initiating (Strict Mode)...")

        # Clean tools: Strip external compatibility keys so OpenAI sees a pure strict request
        clean_tools = [{"type": "function", "function": t["function"]} for t in tools_payload]

        try:
            resp = self.openai.chat.completions.create(
                model="gpt-4.1-mini-2025-04-14",
                messages=messages_payload,
                tools=clean_tools,
                parallel_tool_calls=False  # Strict mode works best with sequential logic
            )
            usage = getattr(resp, "usage", None)
            in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
            out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
            log.info(f"  -> Usage: {in_tok} in / {out_tok} out")
            return resp, in_tok, out_tok

        except Exception as e:
            log.warning(f"OpenAI call failed ({label}): {e}. Attempting sanitized retry.")
            sanitized = self.build_sanitized_messages(messages_payload, content_limit=2000)

            resp2 = self.openai.chat.completions.create(
                model="gpt-4.1-mini-2025-04-14",
                messages=sanitized,
                tools=clean_tools,
                parallel_tool_calls=False
            )
            usage2 = getattr(resp2, "usage", None)
            in_tok2 = getattr(usage2, "prompt_tokens", 0) if usage2 else 0
            out_tok2 = getattr(usage2, "completion_tokens", 0) if usage2 else 0
            log.info(f"  -> Retry Usage: {in_tok2} in / {out_tok2} out")
            return resp2, in_tok2, out_tok2

    async def initialize_guards_and_planner(self):
        await self.planner_adapter.initialize()

        guard_impl = None
        if PaperOpenAIGuard is not None:
            try:
                try:
                    guard_impl = PaperOpenAIGuard(mcp_server_url=None, api_key=OPENAI_API_KEY)
                except TypeError:
                    try:
                        guard_impl = PaperOpenAIGuard(api_key=OPENAI_API_KEY)
                    except Exception:
                        guard_impl = PaperOpenAIGuard(mcp_server_url=None)
            except Exception:
                log.exception("Failed to instantiate PaperOpenAIGuard; guard disabled.")
                guard_impl = None

        self.guard_adapter = GuardAdapter(guard_impl, api_key=OPENAI_API_KEY)
        await self.guard_adapter.initialize(self.sessions)

        if guard_impl is not None:
            try:
                all_tools = await get_mcp_tools_from_sessions(self.sessions)
                # Guard still needs root-level keys
                setattr(guard_impl, "tool_specs", [{"name": t["name"], "description": t["description"],
                                                    "input_schema": t.get("input_schema",
                                                                          t.get("function", {}).get("parameters"))} for
                                                   t in all_tools])
                log.info(f"Guard seeded with {len(getattr(guard_impl, 'tool_specs', []))} aggregated tools.")
            except Exception:
                log.exception("Failed to seed guard with aggregated tools (ignored).")

    async def process_query(self, query: str):
        log.info("--------------------------------------------------")
        log.info(f"New query: '{query}'")

        messages: List[Dict[str, Any]] = []
        total_input_tokens = 0
        total_output_tokens = 0

        user_msg = {"role": "user", "content": query}
        messages.append(user_msg)

        all_tools = await get_mcp_tools_from_sessions(self.sessions)
        pretty_print_tools_plain(all_tools)

        # Planner needs root-level keys
        planner_specs = [{"name": t["name"], "description": t["description"]} for t in all_tools]

        # PLANNER: build route
        route, p_in, p_out = await self.planner_adapter.build_route(query, planner_specs)
        log.info(f"[Planner] Route: {route} (in={p_in} out={p_out})")
        total_input_tokens += int(p_in)
        total_output_tokens += int(p_out)

        class RoutePointer:
            def __init__(self, route_list: List[str]):
                self.route = route_list or []
                self.step = 0

            def current(self):
                if self.step < len(self.route):
                    return self.route[self.step]
                return None

            def advance(self):
                self.step += 1

        pointer = RoutePointer(route)

        step_idx = 0
        assistant_outputs: List[str] = []

        while True:
            allowed_name = pointer.current()
            if allowed_name:
                allowed_tools_payload = [t for t in all_tools if t["name"] == allowed_name]
                log.info(f"Step {step_idx}: allowing tool -> {allowed_name}")
                pretty_print_tools_plain(allowed_tools_payload)
            else:
                allowed_tools_payload = []
                log.info(f"Step {step_idx}: no tool allowed")

            # EXECUTE STEP
            try:
                resp, in_tok, out_tok = self.openai_call_with_retry(messages, allowed_tools_payload,
                                                                    label=f"step_{step_idx}_call")
            except Exception as e:
                log.error(f"OpenAI failure at step {step_idx}: {e}")
                break

            try:
                assistant_msg = resp.choices[0].message
            except Exception:
                choices = getattr(resp, "choices", []) or []
                assistant_msg = choices[0].get("message") if choices else {"content": ""}

            assistant_content = getattr(assistant_msg, "content", None) or (
                assistant_msg.get("content") if isinstance(assistant_msg, dict) else "")

            # Update totals with EXACT usage from API
            total_input_tokens += in_tok
            total_output_tokens += out_tok
            log.info(f"AI step reply (Exact): {in_tok} in / {out_tok} out")

            assistant_entry = {"role": "assistant", "content": assistant_content}
            tool_calls = getattr(assistant_msg, "tool_calls", None)
            if tool_calls is None and isinstance(assistant_msg, dict):
                tool_calls = assistant_msg.get("tool_calls", [])
            tool_calls = tool_calls or []

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
            messages.append(assistant_entry)

            if not tool_calls:
                assistant_outputs.append(assistant_content)
                log.info("No tool calls from assistant — finishing.")
                break

            # Process tool calls
            for tc in tool_calls:
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

                if not full_name:
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": "Error: malformed tool call."})
                    continue

                # Check against route
                expected = pointer.current()
                if expected is None or full_name != expected:
                    log.warning(f"ToolCall {full_name} not allowed/expected ({expected}). Rejecting.")
                    messages.append({"role": "tool", "tool_call_id": tc_id,
                                     "content": f"Error: tool {full_name} not allowed at this step."})
                    continue

                args = {}
                if raw_args:
                    try:
                        # OpenAI Strict Mode guarantees valid JSON
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        args = {"_raw": str(raw_args)}

                # GUARD check
                approved, g_in, g_out, g_raw = True, 0, 0, "PASS (no guard)"
                if self.guard_adapter:
                    try:
                        approved, g_in, g_out, g_raw = await self.guard_adapter.check(full_name, args)
                    except Exception as e:
                        log.exception(f"Guard error: {e}")
                        approved, g_in, g_out, g_raw = False, 0, 0, f"FAIL: {e}"

                # Account Guard tokens
                total_input_tokens += int(g_in)
                total_output_tokens += int(g_out)

                if not approved:
                    log.warning(f"Guard rejected {full_name}: {g_raw}")
                    # --- UPDATE: Tell AI to retry on Guard fail ---
                    messages.append(
                        {"role": "tool", "tool_call_id": tc_id, "content": f"Error: Guard rejected tool call: {g_raw}. Please analyze the rejection reason and retry with corrected arguments."})
                    continue

                log.info(f"Guard approved {full_name} (Guard Usage: {g_in}/{g_out})")

                # EXECUTE
                if "_" not in full_name:
                    messages.append(
                        {"role": "tool", "tool_call_id": tc_id, "content": f"Error: malformed name {full_name}."})
                    continue

                service, tool_name = full_name.split("_", 1)
                session = self.sessions.get(service)
                if not session:
                    messages.append(
                        {"role": "tool", "tool_call_id": tc_id, "content": f"Error: service {service} not connected."})
                    continue

                log.info(f"Executing tool: {service}.{tool_name}")
                try:
                    result = await session.call_tool(tool_name, args)
                    sanitized = sanitize_tool_result(result, max_chars=8000)
                    log.info(f"Tool result length: {len(sanitized)} chars")
                    # Advance pointer ONLY on success
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": sanitized})
                    pointer.advance()
                    log.info(f"Advanced plan pointer to step {pointer.step}")

                except Exception as e:
                    # --- UPDATE: Tell AI to retry on Execution fail ---
                    sanitized = f"Tool execution error: {e}. Please analyze the error and retry."
                    log.error(sanitized)
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": sanitized})

            step_idx += 1

        log.info("Agentic loop complete.")
        log.info(f"Total Input Tokens: {total_input_tokens}")
        log.info(f"Total Output Tokens: {total_output_tokens}")
        log.info(f"Grand Total Tokens: {total_input_tokens + total_output_tokens}")

        assistant_texts = [m.get("content", "") for m in messages if m.get("role") == "assistant"]
        final_text = "\n".join(assistant_texts) if assistant_texts else "(no response)"
        return final_text

    async def chat_loop(self):
        print("MCP ToolRouteGuard Client Ready (Strict Mode).\nType a query, 'quit' to exit.\n")
        while True:
            q = await anyio.to_thread.run_sync(input, "\nQuery: ")
            q = q.strip()
            if q.lower() in ("quit", "exit"):
                break
            out = await self.process_query(q)
            print("\nAssistant:\n", out)


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

        await client.initialize_guards_and_planner()
        await client.chat_loop()


if __name__ == "__main__":
    anyio.run(main, backend="trio")