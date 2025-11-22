"""
mcp_toolroute_zero_mem_unified.py

Unified Multi-Server MCP Client (ZERO MEMORY) + ToolPlanner + ToolRouteGuard + OpenAIGuard
- Zero-memory per query (history resets each query)
- Planner builds a route (OpenAI)
- Guard (OpenAIGuard) validates each tool call (OpenAI)
- Planner & Guard token accounting included in final summary
- Plain-English tool listing (no JSON dumps)
- Real token counting via tiktoken
- Sanitized tool results
- Retry-on-400 sanitized messages preserving required fields
- Safe shutdown wrapper

Usage:
    python mcp_toolroute_zero_mem_unified.py


extract data from this https://www.youtube.com/watch?v=3tEdQBA84tA and this https://www.youtube.com/watch?v=8AxcGiy-RIw video
"""

import os
import json
import logging
import anyio
import tiktoken
from typing import Dict, Any, List, Optional
from contextlib import AsyncExitStack, asynccontextmanager

from dotenv import load_dotenv
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from openai import OpenAI

# ---------------------------------------------------------------------
# CONFIG: MCP SERVERS (tweak as needed)
# ---------------------------------------------------------------------
KLAVIS_SERVERS = {
    # uncomment or add servers you have running
    "gmail": "https://gmail-mcp-server.klavis.ai/mcp/?instance_id=e0ddd5ee-45fc-4791-b9f7-5a04d8a58463",
    #"github": "https://strata.klavis.ai/mcp/?instance_id=df9ad3af-9eb3-4287-b2f4-acbaa5db1138",
    #"linear": "https://linear-mcp-server.klavis.ai/mcp/?instance_id=8e711cd1-909a-4641-95e7-b3d5ee358110",
    #"gcalendar": "https://gcalendar-mcp-server.klavis.ai/mcp/?instance_id=9d9a4b34-d0c5-4b8e-b633-1aa101f57de6",
    #"gdrive": "https://gdrive-mcp-server.klavis.ai/mcp/?instance_id=ab124495-a682-407e-b9a2-d82bb8ab77d0",
    #"jira": "https://strata.klavis.ai/mcp/?instance_id=1c92c41b-f007-4c4b-81ab-be0a6be9b0aa",
    #"notion": "https://strata.klavis.ai/mcp/?instance_id=35468960-6bec-4581-b141-dccd41e87742",
    #"slack": "https://slack-mcp-server.klavis.ai/mcp/?instance_id=bfb2bca1-e73b-4c9d-9338-de1ecd35f4ea",
    #"attio": "https://attio-mcp-server.klavis.ai/mcp/?instance_id=82c94b49-2981-4ab2-9433-1f081a36f22c",
    "hackerNews": "https://hacker-news-mcp-server.klavis.ai/mcp/?instance_id=000c2de5-7296-4417-97e2-6082e77a0050",
    "youtube": "https://youtube-mcp-server.klavis.ai/mcp/?instance_id=e5f0b026-f5db-402e-a6b6-62000d3444ab"
}

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini-2025-04-14")
PLANNER_MODEL = os.getenv("PLANNER_MODEL", OPENAI_MODEL)
GUARD_MODEL = os.getenv("GUARD_MODEL", OPENAI_MODEL)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("MCP-UNIFIED")

# ---------------------------------------------------------------------
# Safe cleanup wrapper
# ---------------------------------------------------------------------
@asynccontextmanager
async def safe_stack():
    try:
        yield
    except Exception:
        # suppress shutdown-related exceptions (task-group race conditions)
        pass

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
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
    """Print tools in readable plain-English lines (no JSON)."""
    if not tools:
        log.info("No tools available.")
        return
    log.info(f"Tools available to AI: {len(tools)}")
    for idx, tf in enumerate(tools, start=1):
        func = tf.get("function", {})
        name = func.get("name", "<unknown>")
        desc = func.get("description", "")
        params = func.get("parameters", None)
        log.info(f"  {idx}. {name}")
        if desc:
            desc_line = "    Desc: " + (desc.replace("\n", " ")[:300] + ("..." if len(desc) > 300 else ""))
            log.info(desc_line)
        if params:
            try:
                if isinstance(params, dict) and "properties" in params:
                    props = params.get("properties", {})
                    if props:
                        param_list = ", ".join([f"{k}" for k in props.keys()])
                        log.info(f"    Params: {param_list}")
                    else:
                        log.info("    Params: (schema present, no properties listed)")
                else:
                    p_short = safe_json_dump(params, max_len=300)
                    log.info(f"    Params: {p_short}")
            except Exception:
                log.info("    Params: (could not parse)")

# ---------------------------------------------------------------------
# ToolPlanner (counts tokens too)
# ---------------------------------------------------------------------
class ToolPlanner:
    """
    Simple synchronous planner using OpenAI client (same SDK as main).
    Returns list of tool names in order.
    Also returns token accounting for the planner call.
    """
    def __init__(self, openai_client: OpenAI, model: str = PLANNER_MODEL):
        self.openai = openai_client
        self.model = model
        # tiktoken for planner
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def _build_prompt(self, user_query: str, tool_specs: List[Dict[str, Any]]) -> str:
        tool_list_text = "\n".join(f"- {t['name']}: {t.get('description','')}" for t in tool_specs)
        prompt = f"""
You are a planner that returns a single Python list of tools (exact names) to satisfy the user's request.

User request:
---
{user_query}
---

Available tools (name: description):
{tool_list_text}

Rules:
1) Return ONLY a valid Python list of tool names (strings) in the exact order to execute.
2) Use only tools from the available list.
3) If no tool is needed, return [].
4) Do NOT add commentary, explanations or any other text.
5) Example: ["gcalendar_list_events", "youtube_get_transcript", "gmail_send_email"]

Return the list now.
"""
        return prompt

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text or ""))

    def count_payload_tokens(self, messages: List[Dict[str, Any]]) -> int:
        payload = {"model": self.model, "messages": messages}
        txt = json.dumps(payload, default=str, ensure_ascii=False)
        return len(self.encoding.encode(txt))

    def build_route(self, user_query: str, tool_specs: List[Dict[str, Any]]) -> (List[str], int, int):
        """
        Returns (route_list, input_tokens_estimate, output_tokens_estimate)
        """
        prompt = self._build_prompt(user_query, tool_specs)
        messages = [{"role": "user", "content": prompt}]
        input_tokens = self.count_payload_tokens(messages)
        log.info(f"[Planner] Input tokens ≈ {input_tokens}")

        try:
            resp = self.openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0
            )
        except Exception as e:
            log.error(f"[Planner] OpenAI error: {e}")
            return [], input_tokens, 0

        # defensive extract
        try:
            text = resp.choices[0].message.content.strip()
        except Exception:
            choices = getattr(resp, "choices", []) or []
            text = choices[0].get("message", {}).get("content", "") if choices else ""

        out_tokens = self.count_tokens(text)
        log.info(f"[Planner] Output tokens ≈ {out_tokens}")
        log.info(f"[Planner] Raw output (truncated): {text[:1000]}")

        # parse into list
        route = []
        try:
            parsed = eval(text, {"__builtins__": None}, {})
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                route = parsed
            else:
                log.warning("[Planner] Parsed response not a list of strings. Returning empty route.")
        except Exception:
            log.warning("[Planner] Could not parse planner output. Returning empty route.")

        return route, input_tokens, out_tokens

# ---------------------------------------------------------------------
# Guard (OpenAI-based) - validates a single tool call and counts tokens
# ---------------------------------------------------------------------
class OpenAIGuard:
    """
    Guard that checks tool usage via OpenAI. Returns PASS/FAIL and token accounting.
    """
    def __init__(self, mcp_tool_specs: List[Dict[str, Any]], model: str = GUARD_MODEL, openai_client: Optional[OpenAI] = None):
        self.tool_specs = mcp_tool_specs  # expected list of dicts with name, description, input_schema
        self.model = model
        self.openai = openai_client or OpenAI()
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text or ""))

    def count_payload_tokens(self, messages: List[Dict[str, Any]]) -> int:
        payload = {"model": self.model, "messages": messages}
        txt = json.dumps(payload, default=str, ensure_ascii=False)
        return len(self.encoding.encode(txt))

    def _build_prompt(self, tool_name: str, tool_info: Dict[str, Any], input_data: Dict[str, Any]) -> str:
        schema_str = safe_json_dump(tool_info.get("input_schema", {}), max_len=2000)
        prompt = f"""
You are a tool-validation AI.

Tool Name: {tool_name}
Description: {tool_info.get('description', '')}
Input Schema:
{schema_str}
Proposed Input:
{safe_json_dump(input_data, max_len=2000)}

Respond with ONLY PASS or FAIL (single word).
"""
        return prompt

    def check(self, tool_name: str, input_data: Dict[str, Any]) -> (bool, int, int, str):
        """
        Synchronous check. Returns (approved_bool, input_tokens, output_tokens, raw_text)
        """
        tool_info = next((t for t in self.tool_specs if t["name"] == tool_name), None)
        if not tool_info:
            log.warning(f"[Guard] Unknown tool requested: {tool_name}")
            return False, 0, 0, "FAIL (unknown tool)"

        prompt = self._build_prompt(tool_name, tool_info, input_data)
        messages = [{"role": "user", "content": prompt}]
        in_toks = self.count_payload_tokens(messages)
        log.info(f"[Guard] Input tokens ≈ {in_toks} (for tool {tool_name})")

        try:
            resp = self.openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0
            )
        except Exception as e:
            log.error(f"[Guard] OpenAI error: {e}")
            return False, in_toks, 0, "FAIL (openai error)"

        try:
            text = resp.choices[0].message.content.strip()
        except Exception:
            choices = getattr(resp, "choices", []) or []
            text = choices[0].get("message", {}).get("content", "") if choices else ""

        out_toks = self.count_tokens(text)
        log.info(f"[Guard] Output tokens ≈ {out_toks} (raw: {text.strip()[:200]})")

        normalized = text.upper().replace('"', "").replace("'", "").strip()
        approved = normalized.startswith("PASS")
        return approved, in_toks, out_toks, text

# ---------------------------------------------------------------------
# Utilities: get tools from sessions
# ---------------------------------------------------------------------
async def get_mcp_tools_from_sessions(sessions: Dict[str, ClientSession]) -> List[Dict[str, Any]]:
    """
    Query each connected session for its tools and return a flattened list of tool specs
    in the OpenAI 'function' format we use (plus convenience keys).
    Each entry includes:
      { "type":"function", "function": {...}, "name": "<service>_<toolname>", "input_schema": ..., "description": ... }
    """
    all_tools = []
    for srv_name, session in sessions.items():
        try:
            resp = await session.list_tools()
            server_tools = getattr(resp, "tools", []) or []
            for t in server_tools:
                func_name = f"{srv_name}_{t.name}"
                entry = {
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "description": getattr(t, "description", "") or "",
                        "parameters": getattr(t, "inputSchema", None)
                    },
                    # convenience top-level fields for planner/guard
                    "name": func_name,
                    "description": getattr(t, "description", "") or "",
                    "input_schema": getattr(t, "inputSchema", None)
                }
                all_tools.append(entry)
        except Exception as e:
            log.warning(f"Could not list tools from {srv_name}: {e}")
    return all_tools

# ---------------------------------------------------------------------
# MCP Client (zero memory + planner + guard integrated)
# ---------------------------------------------------------------------
class MultiMCPClient:
    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.openai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()
        # tiktoken encoder for main client counting
        try:
            self.encoding = tiktoken.encoding_for_model(OPENAI_MODEL)
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_payload_tokens(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> int:
        payload = {"model": OPENAI_MODEL, "messages": messages, "tools": tools}
        text = json.dumps(payload, default=str, ensure_ascii=False)
        return len(self.encoding.encode(text))

    def count_output_tokens(self, text: Optional[str]) -> int:
        return len(self.encoding.encode(text or ""))

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
                        tc_list.append({"id": tc_id, "type": "function", "function": {"name": fname, "arguments": args_str}})
                    except Exception:
                        tc_list.append({"id": None, "type": "function", "function": {"name": None, "arguments": ""}})
                entry["tool_calls"] = tc_list
            sanitized.append(entry)
        return sanitized

    def openai_call_with_retry(self, messages_payload: List[Dict[str, Any]], tools_payload: List[Dict[str, Any]], label: str = "initial"):
        sent_tokens = self.count_payload_tokens(messages_payload, tools_payload)
        log.info(f"{label.capitalize()} call → approx {sent_tokens} tokens (payload)")
        try:
            resp = self.openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages_payload,
                tools=tools_payload
            )
            return resp, sent_tokens
        except Exception as e:
            log.warning(f"OpenAI call failed ({label}): {e}. Attempting sanitized retry.")
            sanitized = self.build_sanitized_messages(messages_payload, content_limit=2000)
            sent2 = self.count_payload_tokens(sanitized, tools_payload)
            log.info(f"Sanitized retry → approx {sent2} tokens")
            resp2 = self.openai.chat.completions.create(
                model=OPENAI_MODEL,
                messages=sanitized,
                tools=tools_payload
            )
            return resp2, sent2

    async def process_query(self, query: str) -> str:
        log.info("--------------------------------------------------")
        log.info(f"New query: {query!r}")

        # ZERO MEMORY: reset per-query state and token tallies
        messages: List[Dict[str, Any]] = []
        total_input_tokens = 0
        total_output_tokens = 0

        # append only the current user message
        user_msg = {"role": "user", "content": query}
        messages.append(user_msg)

        # fetch all tools for planner & mapping
        all_tools = await get_mcp_tools_from_sessions(self.sessions)
        pretty_print_tools_plain(all_tools)

        # Build planner-friendly specs (name + description)
        tool_specs_for_planner = [{"name": t["name"], "description": t["description"]} for t in all_tools]

        # Planner: build route and collect planner token counts
        planner = ToolPlanner(self.openai, model=PLANNER_MODEL)
        route, planner_in_toks, planner_out_toks = planner.build_route(query, tool_specs_for_planner)
        log.info(f"[Planner] Route: {route}")
        total_input_tokens += planner_in_toks
        total_output_tokens += planner_out_toks

        # Prepare guard with tool metadata (names + schemas)
        guard = OpenAIGuard(mcp_tool_specs=[{"name": t["name"], "description": t["description"], "input_schema": t.get("input_schema", t.get("function", {}).get("parameters"))} for t in all_tools], model=GUARD_MODEL, openai_client=self.openai)

        # wrap route in simple guard container (index pointer)
        class RoutePointer:
            def __init__(self, route_list: List[str]):
                self.route = route_list or []
                self.step = 0
            def current(self) -> Optional[str]:
                if self.step < len(self.route):
                    return self.route[self.step]
                return None
            def advance(self):
                self.step += 1

        pointer = RoutePointer(route)

        # If planner returned no tools, we still call model once with no tools (it can answer directly)
        step_idx = 0
        assistant_output_accum: List[str] = []

        while True:
            # Determine allowed tool for this step (only one to expose)
            allowed_tool_name = pointer.current()
            if allowed_tool_name:
                allowed_tools_payload = [t for t in all_tools if t["name"] == allowed_tool_name]
                log.info(f"Step {step_idx}: allowing tool -> {allowed_tool_name}")
                pretty_print_tools_plain(allowed_tools_payload)
            else:
                allowed_tools_payload = []
                log.info(f"Step {step_idx}: no tool allowed (plan exhausted or none required)")

            # call model with current messages and allowed tools
            try:
                resp, sent = self.openai_call_with_retry(messages, allowed_tools_payload, label=f"step_{step_idx}_call")
            except Exception as e:
                log.error(f"OpenAI failure at step {step_idx}: {e}")
                break

            # extract assistant message & tokens
            try:
                assistant_msg = resp.choices[0].message
            except Exception:
                choices = getattr(resp, "choices", []) or []
                assistant_msg = choices[0].get("message") if choices else {"content": ""}

            assistant_content = getattr(assistant_msg, "content", None) or (assistant_msg.get("content") if isinstance(assistant_msg, dict) else "")
            out_tokens = self.count_output_tokens(assistant_content)
            log.info(f"AI replied using ~{out_tokens} tokens (assistant output)")

            total_input_tokens += sent
            total_output_tokens += out_tokens

            # append assistant entry
            assistant_entry = {"role": "assistant", "content": assistant_content}
            # preserve tool_calls if present (minimal)
            tool_calls = getattr(assistant_msg, "tool_calls", None)
            if tool_calls is None and isinstance(assistant_msg, dict):
                tool_calls = assistant_msg.get("tool_calls", [])
            tool_calls = tool_calls or []
            if tool_calls:
                # normalize
                safe_calls = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        tc_copy = dict(tc)
                        if "type" not in tc_copy:
                            tc_copy["type"] = "function"
                        safe_calls.append(tc_copy)
                    else:
                        func = getattr(tc, "function", None)
                        tc_id = getattr(tc, "id", None)
                        fname = getattr(func, "name", None) if func else None
                        args = getattr(func, "arguments", None) if func else None
                        safe_calls.append({"id": tc_id, "type": "function", "function": {"name": fname, "arguments": args}})
                assistant_entry["tool_calls"] = safe_calls
            messages.append(assistant_entry)

            # If no tool calls -> finish
            if not tool_calls:
                assistant_output_accum.append(assistant_content)
                log.info("No tool calls from assistant — finishing.")
                break

            # Process each tool call emitted by the model
            for tc in tool_calls:
                # extract full name + args
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
                    log.warning("Malformed tool call (no function name). Appending error to messages.")
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": f"Error: malformed tool call (no name)."})
                    continue

                # Guard check: ensure matches planned route current step
                expected = pointer.current()
                if expected is None:
                    log.warning(f"ToolCall {full_name} not allowed (no expected tool). Rejecting.")
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": f"Error: tool {full_name} not allowed at this step."})
                    continue
                if full_name != expected:
                    log.warning(f"ToolCall {full_name} does not match expected {expected}. Rejecting.")
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": f"Error: tool {full_name} not allowed at this step (expected {expected})."})
                    continue

                # parse args safely
                args = {}
                if raw_args:
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        args = {"_raw": str(raw_args)}

                # Run guard validation (counts tokens for guard)
                approved, guard_in_toks, guard_out_toks, guard_raw = guard.check(full_name, args)
                total_input_tokens += guard_in_toks
                total_output_tokens += guard_out_toks
                if not approved:
                    log.warning(f"[Guard] Rejected tool call {full_name}. Appending rejection to messages.")
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": f"Error: Guard rejected tool call: {guard_raw}"})
                    # don't advance pointer — model could try alternate or we stop
                    continue
                else:
                    log.info(f"[Guard] Approved tool call {full_name} (guard tokens: in={guard_in_toks} out={guard_out_toks})")

                # find the session and run the tool
                if "_" not in full_name:
                    log.warning(f"Tool name {full_name} missing service prefix. Skipping.")
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": f"Error: malformed tool name {full_name}."})
                    continue

                service, tool_name = full_name.split("_", 1)
                session = self.sessions.get(service)
                if not session:
                    log.warning(f"Service {service} not connected for tool {full_name}.")
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": f"Error: service {service} not connected."})
                    continue

                log.info(f"Executing tool: {service}.{tool_name} (id={tc_id})")
                try:
                    result = await session.call_tool(tool_name, args)
                    sanitized = sanitize_tool_result(result, max_chars=8000)
                    log.info(f"Tool result length: {len(sanitized)} chars")
                except Exception as e:
                    sanitized = f"Tool execution error: {e}"
                    log.error(sanitized)

                # append tool result for model to see in next loop
                messages.append({"role": "tool", "tool_call_id": tc_id, "content": sanitized})

                # if guard & execution successful, advance pointer to next plan item
                pointer.advance()
                log.info(f"Advanced plan pointer to step {pointer.step}")

            # loop to next step (model will be called again)
            step_idx += 1

        # done - final summary with token breakdown
        log.info("Agentic loop complete.")
        log.info(f"Planner tokens (in+out) ≈ {planner_in_toks + planner_out_toks}")
        # total_input_tokens/total_output_tokens include planner & guard counts already aggregated
        log.info(f"Input tokens total (approx): {total_input_tokens}")
        log.info(f"Output tokens total (approx): {total_output_tokens}")
        log.info(f"Combined tokens (approx): {total_input_tokens + total_output_tokens}")

        # return assistant aggregated outputs (or last assistant content)
        assistant_texts = [m.get("content", "") for m in messages if m.get("role") == "assistant"]
        final_text = "\n".join(assistant_texts) if assistant_texts else "(no response)"
        return final_text

    async def chat_loop(self):
        print("MCP ToolRouteGuard Unified Client Ready.\nType a query, 'quit' to exit.\n")
        while True:
            q = await anyio.to_thread.run_sync(input, "\nQuery: ")
            q = q.strip()
            if q.lower() in ("quit", "exit"):
                break
            out = await self.process_query(q)
            print("\nAssistant:\n", out)

# ---------------------------------------------------------------------
# Main: connect to servers and run
# ---------------------------------------------------------------------
async def main():
    client = MultiMCPClient()
    async with safe_stack(), AsyncExitStack() as stack:
        # connect to every server in KLAVIS_SERVERS
        for name, url in KLAVIS_SERVERS.items():
            try:
                streams = await stack.enter_async_context(streamablehttp_client(url))
                session = await stack.enter_async_context(ClientSession(streams[0], streams[1]))
                await session.initialize()
                client.sessions[name] = session
                log.info(f"Connected to {name}")
            except Exception as e:
                log.warning(f"Failed to connect to {name}: {e}")

        await client.chat_loop()

if __name__ == "__main__":
    anyio.run(main, backend="trio")
