# StreamKnight: Stream Tool Injection and Monitoring Technique

**StreamKnight** is a custom guardrail architecture designed for the Model Context Protocol (MCP). It addresses the high token overhead and reliability issues encountered when AI models interact with multiple MCP servers. By implementing a "Stream Tool Injection and Monitoring" technique, StreamKnight constructs explicit tool routes and monitors execution in real-time.

Research Code - research.py
Normal Code - normal.py

## 🚀 Key Features

  * **Token Optimization:** Reduces token consumption by **65% to 88%** depending on the number of MCP servers used.
  * **Context Isolation:** Exposes only the *necessary* tool at each step of execution, preventing context window saturation.
  * **Semantic Guardrails:** Validates tool arguments (schema and semantic checks) before execution to prevent incorrect usage (e.g., stopping the use of placeholder emails like `example@gmail.com`).
  * **Self-Correction:** Provides feedback to the AI model to correct invalid arguments without user intervention.

## 🧠 The Problem

The Model Context Protocol (MCP) standardizes how AI integrates with external tools. However, as the number of MCP servers increases, two critical issues emerge:

1.  **Token Overhead:** Feeding the entire tool inventory (names, descriptions, schemas) to the LLM for every turn is computationally expensive and fills the context window.
2.  **Execution Reliability:** Models occasionally hallucinate tool arguments or use incorrect formats, leading to failed execution.

## 🛠️ Architecture

StreamKnight utilizes a three-phase methodology to solve these problems:

### Phase 1: Trajectory Definition (Tool Planner)

A dedicated "Tool Planner" AI scans the tool inventory once and generates a deterministic route of tools required to solve the user query. This avoids passing massive tool definitions to the main execution agent repeatedly.

**Example Route:**

```json
["github_search_repos", "github_get_repo_details", "linear_create_issue"]
```

### Phase 2: Orchestration Loop (Context Isolation)

Using a `RoutePointer` mechanism, the system enforces strict context isolation. At any given step `N`, the AI agent can **only** see the definition for the tool assigned to step `N`.

```python
# Simplified Logic
class RoutePointer:
    def current(self) -> Optional[str]:
        """Returns the currently allowed tool name"""
        return self.route[self.step]
```

### Phase 3: Semantic Guardrail

Before any tool is executed, a "Guard AI" validates the call. It checks:

  * **Schema Compliance:** Do arguments match the expected format?
  * **Semantic Validity:** Do the values make sense in context?

If a check fails, the Guard returns a `FAIL` verdict with a specific reason, triggering a self-correction loop.

## 📊 Performance Results

We evaluated StreamKnight against a standard MCP client using 11 different MCP servers (88 total tools), including GitHub, Slack, Gmail, and Linear.

### Token Reduction

As the number of MCP servers increases, the efficiency of StreamKnight improves significantly compared to standard execution.

| MCP Servers | Token Reduction |
| :--- | :--- |
| 0 Servers | **65%** |
| 1 Server | **77%** |
| 2 Servers | **84%** |
| 3 Servers | **88%** |

### Reliability

In controlled tests where models were intentionally instructed to use wrong arguments, the Guard architecture successfully detected and rejected **100%** of incorrect tool calls.

## 💻 Tech Stack & Experimental Setup

  * **Language:** Python 3.11
  * **Model Engine:** GPT-4.1-mini (utilized for Planner, Guard, and Main AI)
  * **Tools:** Klavis AI MCP Servers (GitHub, Attio, Slack, YouTube, HackerNews, Gmail, Linear, Google Calendar, Jira, Notion, Google Drive)

## 🔗 Resources

  * **Paper/Repository:** [StreamKnight GitHub](https://github.com/StreamKnight/StreamKnight-Guardrails/tree/main/paper)
  * **Evaluation Data:** [Raw Data Sheet](https://docs.google.com/spreadsheets/d/122D9rTQx44gKXrZxSSvdgm4wLoiyUBNopyGlkK761Y8/edit?usp=sharing)

## 👥 Authors:

  * **Mayank Singh Jadon** - mayank.msj.singh@gmail.com
  * **Krishna Nishad** - krishnanishad52513@gmail.com
  * **Akshay Tiwari** - akshay2003tiwari@gmail.com
  * **Amrita Bhatnagar (Supervisor)** - bhatnagaramrita@akgec.ac.in
