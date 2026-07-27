"""
Local natural-language agent for Windows, powered by Ollama + MCP servers.

This extends agent.py: in addition to the hand-written local tools
(open_url, open_app, create_github_repo, list_directory), it connects to
any MCP servers you configure below, pulls in their tools automatically,
and lets the model call them the same way.

Requirements:
  pip install ollama mcp

  Node.js installed (most community MCP servers run via `npx`)
  https://nodejs.org

Run:
  python agent_mcp.py
"""

import asyncio
import json
import os
import subprocess
import sys
import webbrowser

# Force UTF-8 on stdout/stderr so tool snapshots (which include arrows, em
# dashes, etc.) don't crash the cp1252 console on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from contextlib import AsyncExitStack

import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MODEL = "qwen2.5:7b"

# ---------------------------------------------------------------------------
# Configure MCP servers here. Each one is a separate process the script
# launches and talks to over stdio. Scope filesystem/github servers to only
# what you want the agent touching.
# ---------------------------------------------------------------------------

MCP_SERVERS = [
    # {
    #     "name": "github",
    #     "command": "npx",
    #     "args": ["-y", "@modelcontextprotocol/server-github"],
    #     "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_xxx"},
    # },
        {
        # Drives the user's already-running Opera over the Chrome DevTools
        # Protocol via our own Python MCP server (opera_cdp_mcp.py). This
        # is a custom, minimal server that bypasses the @playwright/mcp
        # alpha handshake which hangs against Opera 133 / Chrome 149.
        #
        # Requirements:
        #   1. Opera must be running with --remote-debugging-port=9222.
        #      The Start Menu shortcut has been patched to add this flag
        #      so just launch Opera normally.
        #   2. This MCP server connects to that port and drives the same
        #      tabs the user has open — no new browser process is
        #      spawned, no extra Opera window appears, and your existing
        #      logged-in sessions (chess.com, gmail, etc.) are reused.
        "name": "opera",
        "command": sys.executable,
        "args": [os.path.join(os.path.dirname(os.path.abspath(__file__)), "opera_cdp_mcp.py")],
    },
]

# ---------------------------------------------------------------------------
# Local (non-MCP) tools — same as before
# ---------------------------------------------------------------------------

def open_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    webbrowser.open(url)
    return f"Opened {url} in your default browser."


def open_app(name: str) -> str:
    known_apps = {
        "notepad": "notepad.exe", "calculator": "calc.exe", "calc": "calc.exe",
        "explorer": "explorer.exe", "paint": "mspaint.exe", "cmd": "cmd.exe",
        "terminal": "wt.exe", "task manager": "taskmgr.exe",
        "control panel": "control.exe", "vscode": "code", "vs code": "code",
        "word": "winword.exe", "excel": "excel.exe",
    }
    target = known_apps.get(name.lower(), name)
    try:
        os.startfile(target)
        return f"Opened {name}."
    except FileNotFoundError:
        try:
            subprocess.Popen(target, shell=True)
            return f"Launched {name}."
        except Exception as e:
            return f"Couldn't open '{name}': {e}"


def create_github_repo(name: str, private: bool = False) -> str:
    visibility = "--private" if private else "--public"
    try:
        result = subprocess.run(
            ["gh", "repo", "create", name, visibility, "--confirm"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return f"Created GitHub repo '{name}'.\n{result.stdout.strip()}"
        return f"gh reported an error: {result.stderr.strip()}"
    except FileNotFoundError:
        return "GitHub CLI ('gh') isn't installed."


def list_directory(path: str = ".") -> str:
    try:
        entries = os.listdir(path)
        return "\n".join(entries) if entries else "(empty directory)"
    except Exception as e:
        return f"Couldn't list '{path}': {e}"


LOCAL_TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "open_url", "description": "Open a website URL in the default browser.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "open_app", "description": (
            "Open a LOCAL DESKTOP PROGRAM already installed on this Windows PC (e.g. "
            "notepad, calculator, vscode), or a local file/folder path. "
            "Never use this for websites or URLs (chess.com, youtube.com, gmail, etc) "
            "even if they mention a well-known site name that sounds like an app "
            "— use the browser navigation tool for anything web-based instead."),
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "create_github_repo", "description": "Create a new GitHub repository via gh CLI.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}, "private": {"type": "boolean"}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "list_directory", "description": "List files in a directory.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": []}}},
]

LOCAL_DISPATCH = {
    "open_url": open_url,
    "open_app": open_app,
    "create_github_repo": create_github_repo,
    "list_directory": list_directory,
}


class MCPManager:
    """Connects to configured MCP servers and routes tool calls to them."""

    def __init__(self):
        self.stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}   # tool_name -> session
        self.tool_specs: list[dict] = []

    async def connect_all(self):
        for server in MCP_SERVERS:
            params = StdioServerParameters(
                command=server["command"], args=server["args"], env=server.get("env"),
            )
            read, write = await self.stack.enter_async_context(stdio_client(params))
            session = await self.stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                self.sessions[tool.name] = session
                self.tool_specs.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema,
                    },
                })
            print(f"  connected: {server['name']} ({len(tools_result.tools)} tools)")

    async def call(self, name: str, args: dict) -> str:
        session = self.sessions.get(name)
        if session is None:
            return f"Unknown MCP tool: {name}"
        result = await session.call_tool(name, args)
        parts = [c.text for c in result.content if hasattr(c, "text")]
        return "\n".join(parts) if parts else str(result.content)

    async def close(self):
        await self.stack.aclose()


async def run_agent():
    mcp = MCPManager()
    print("Connecting to MCP servers...")
    if MCP_SERVERS:
        await mcp.connect_all()
    else:
        print("  (none configured)")

    all_tool_specs = LOCAL_TOOL_SPECS + mcp.tool_specs
    mcp_tool_names = {spec["function"]["name"] for spec in mcp.tool_specs}

    print(f"\nLocal agent ready (model: {MODEL}). Type a request, or 'exit' to quit.\n")

    messages = [{
        "role": "system",
        "content": (
            "You are a local assistant that controls the user's Windows computer "
            "through the provided tools, including any connected MCP tools. "
            "Always use a tool when the user asks you to open, create, read, or "
            "modify something. "
            "For ANY website, URL, or web service (chess.com, youtube, gmail, etc), "
            "always use the browser/Opera MCP tools (browser_navigate, "
            "browser_snapshot, browser_click, browser_type, browser_press_key, "
            "browser_evaluate) — never open_app, which is only for local desktop "
            "programs already installed on this PC. "
            "\n"
            "BROWSER TOOLS — what they actually do:\n"
            "  browser_navigate(url)   opens a new tab in the user's running "
            "Opera and navigates to the URL. Returns a snapshot.\n"
            "  browser_snapshot()      re-reads the current tab and lists every "
            "interactive element with a stable CSS selector you can paste into "
            "browser_click / browser_type.\n"
            "  browser_click(selector) scrolls the element into view, then "
            "dispatches a real mouse click at its center.\n"
            "  browser_type(selector, text) focuses the input and types each "
            "character.\n"
            "  browser_press_key(key)  one keypress (Enter, Escape, ArrowDown, "
            "Tab, /, etc).\n"
            "  browser_tabs(action)    list/open/select/close tabs.\n"
            "  browser_evaluate(expr)  run JS on the current tab, return value.\n"
            "\n"
            "BROWSER DISCIPLINE (most important):\n"
            "1. NEVER invent page content. Only describe what browser_snapshot "
            "actually returned.\n"
            "2. After every browser_navigate or browser_click, the next tool "
            "call MUST be browser_snapshot — never guess at what a page looks "
            "like without snapshotting it first.\n"
            "3. After receiving a snapshot, your NEXT tool call MUST be a "
            "concrete action (browser_click, browser_type, browser_press_key, "
            "browser_navigate, browser_evaluate) — never just summarize and "
            "stop. The user asked you to *do* something, not describe it.\n"
            "4. The snapshot gives you CSS selectors like "
            "`a[href=\"/play\"]` or `button:nth-of-type(2)`. Copy that exact "
            "selector into browser_click — character for character. Do NOT "
            "make up new selectors, do NOT use `:contains()` (that's jQuery, "
            "not CSS), do NOT use text matches. If a selector from a previous "
            "step is no longer in the latest snapshot, the page changed and "
            "you must use the NEW selector from the fresh snapshot.\n"
            "4b. If after a click the URL contains `base=180` or `time=3`, "
            "the 3-minute match was already created and matchmaking is "
            "running — stop clicking, reply to the user.\n"
            "5. Never guess a deep URL path (e.g. /match-making/3-min). "
            "Navigate to the homepage first, then click through the visible "
            "buttons. Only use a direct URL the user gave you.\n"
            "6. If the snapshot shows a login wall or CAPTCHA, stop and tell "
            "the user — don't try to type credentials on their behalf.\n"
            "7. If a tool call returns the same error twice in a row, stop and "
            "tell the user — don't loop hoping it will start working.\n"
            "\n"
            "WORKED EXAMPLE — 'open chess.com and start a 3 minute match':\n"
            "  step 1: browser_navigate('https://www.chess.com/')\n"
            "  step 2: browser_snapshot  → see the actual buttons; one will "
            "have a selector like `a[href=\"/play\"]`\n"
            "  step 3: browser_click(selector='a[href=\"/play\"]')\n"
            "  step 4: browser_snapshot  → confirm the play page loaded and "
            "see the time controls\n"
            "  step 5: browser_click(selector='<the 3-min button>')\n"
            "  step 6: browser_click(selector='<the Play button>')\n"
            "  step 7: browser_snapshot  → confirm matchmaking is happening\n"
            "  step 8: short reply quoting the final snapshot text.\n"
            "\n"
            "Keep replies short. If a tool result is an error, report the error "
            "honestly instead of claiming the action succeeded."
        ),
    }]

    client = ollama.AsyncClient()

    try:
        while True:
            user_input = input("> ").strip()
            if user_input.lower() in ("exit", "quit"):
                break
            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})

            # Agentic loop: keep letting the model call tools until it stops
            # asking for one. A multi-step browser task (navigate → snapshot →
            # click → snapshot → click → snapshot → click → snapshot → reply)
            # uses ~9 tool calls on its own.
            MAX_STEPS = 20
            for _ in range(MAX_STEPS):
                response = await client.chat(model=MODEL, messages=messages, tools=all_tool_specs)
                msg = response["message"]
                messages.append(msg)

                tool_calls = msg.get("tool_calls")
                if not tool_calls:
                    print(msg.get("content", "(no response)"))
                    break

                for call in tool_calls:
                    fn_name = call["function"]["name"]
                    raw_args = call["function"]["arguments"]
                    args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args)

                    print(f"  [running {fn_name}({args})]")
                    if fn_name in mcp_tool_names:
                        result = await mcp.call(fn_name, args)
                    elif fn_name in LOCAL_DISPATCH:
                        result = LOCAL_DISPATCH[fn_name](**args)
                    else:
                        result = f"Unknown tool: {fn_name}"

                    result_str = str(result)
                    preview = result_str if len(result_str) < 300 else result_str[:300] + "... (truncated)"
                    # Console may be cp1252; force UTF-8 for the preview to keep
                    # the run readable (snapshots contain → and other chars).
                    print(f"    -> {preview}".encode("utf-8", "replace").decode("utf-8", "replace"))

                    messages.append({"role": "tool", "content": result_str})

                    # After any state-changing browser action, also re-read the
                    # page so the model sees the CURRENT DOM (not the snapshot
                    # from the previous step). This is what stops small models
                    # from reusing stale selectors.
                    if fn_name in {"browser_navigate", "browser_click", "browser_type", "browser_press_key", "browser_tabs"} \
                            and "browser_snapshot" in mcp_tool_names:
                        try:
                            snap = await mcp.call("browser_snapshot", {})
                            snap_str = str(snap)
                            print("    [auto-snapshot -> fresh DOM after " + fn_name + "]")
                            messages.append({
                                "role": "tool",
                                "content": "[auto-snapshot after " + fn_name + " — use these selectors, not older ones]\n" + snap_str,
                            })
                        except Exception as e:
                            print(f"    [auto-snapshot ERR {e}]")
            else:
                print(f"  (stopped after {MAX_STEPS} tool steps — task may be incomplete)")
    finally:
        await mcp.close()


if __name__ == "__main__":
    asyncio.run(run_agent())