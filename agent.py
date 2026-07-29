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
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
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


# ---------------------------------------------------------------------------
# Web research tools (stdlib-only — no extra packages)
# ---------------------------------------------------------------------------
#
# These two short-circuit the expensive "navigate → snapshot → type into search
# box → snapshot → click first result → snapshot" path for the very common case
# of "find N sources / papers / docs on X". The model should still fall back to
# the full browser tools when:
#   - the source needs JS to render (e.g. arXiv abstract pages mostly work,
#     but some journal sites are JS-only — fall back to browser_navigate)
#   - the task requires logging in or interacting with the page
#   - web_fetch returns an empty/short body
#

_DDG_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def _http_get(url: str, timeout: float = 20.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _DDG_UA, "Accept-Language": "en-US,en;q=0.9"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        # DuckDuckGo HTML is latin-1-ish; force-decode conservatively.
        raw = r.read()
        charset = r.headers.get_content_charset() or "utf-8"
        try:
            return raw.decode(charset, errors="replace")
        except LookupError:
            return raw.decode("utf-8", errors="replace")


def web_search(query: str, n: int = 8) -> str:
    """Search the web via DuckDuckGo's HTML endpoint and return up to n
    (title, url) pairs. Cheap (one HTTP request) — use this first whenever
    the user asks to "find N sources / papers / articles on X" before
    falling back to driving the browser."""
    n = max(1, min(int(n), 20))
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    try:
        html = _http_get(url, timeout=20)
    except Exception as e:
        return f"web_search failed: {e}"

    # DDG HTML result blocks look like:
    #   <a class="result__a" href="...">TITLE</a>
    #   <a class="result__snippet">SNIPPET</a>
    # Be forgiving — fall back to any <a> with an http(s) href if the
    # class-based selector misses.
    pat_title = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    pat_snip = re.compile(
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )

    def _clean(s: str) -> str:
        s = re.sub(r"<[^>]+>", "", s)
        return re.sub(r"\s+", " ", s).strip()

    titles_urls = pat_title.findall(html)
    snippets = [_clean(s) for s in pat_snip.findall(html)]

    if not titles_urls:
        # Loose fallback: any anchor with an http(s) href and non-empty text.
        loose = re.findall(
            r'<a[^>]+href="(https?://[^"]+)"[^>]*>([^<]{8,200})</a>', html
        )
        titles_urls = [(u, t) for u, t in loose if "duckduckgo" not in u]

    if not titles_urls:
        return "(no results returned by DuckDuckGo)"

    out = [f"Search: {query!r}  ({len(titles_urls)} hits, showing top {n})"]
    for i, (u, t) in enumerate(titles_urls[:n], 1):
        title = _clean(t) or "(no title)"
        out.append(f"{i}. {title}\n   {u}")
        if i - 1 < len(snippets) and snippets[i - 1]:
            out.append(f"   snippet: {snippets[i - 1][:200]}")
    return "\n".join(out)


_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<script\b[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL)
_STYLE_RE = re.compile(r"<style\b[^>]*>.*?</style>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"\s+")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


# ---------------------------------------------------------------------------
# General HTTP tool — covers anything web_search/web_fetch can't:
#   - non-GET methods (POST/PUT/PATCH/DELETE)
#   - custom headers (Authorization, cookies, content-type)
#   - JSON / form / raw bodies
#   - any URL, any port
# Falls back to the browser tools when:
#   - the response is empty/JS-rendered and you need the real DOM
#   - the site requires interactive login flows (CAPTCHA, OAuth consent)
#   - you need cookies from a logged-in session — http_request has none of
#     the user's browser session cookies, so for "post to my X account"
#     type tasks, drive the browser instead.
# ---------------------------------------------------------------------------

# Cap to keep the model from accidentally pulling gigabytes through Ollama's
# tool-result channel. 2 MiB is enough for almost any API response.
_HTTP_MAX_BYTES = 2 * 1024 * 1024


def http_request(
    url: str,
    method: str = "GET",
    headers: dict | None = None,
    body: str | None = None,
    json_body: dict | list | None = None,
    form_body: dict | None = None,
    timeout: float = 30.0,
    follow_redirects: bool = True,
) -> str:
    """Make an HTTP request and return the response.

    Use this for any web task that web_search / web_fetch can't cover:
    calling REST APIs, POSTing a form, hitting a webhook, downloading a
    JSON resource with auth headers, etc.

    Args:
        url: Absolute http(s) URL.
        method: HTTP method (GET, POST, PUT, PATCH, DELETE, ...). Default GET.
        headers: Dict of extra request headers (e.g. {"Authorization":
            "Bearer xyz"}, {"Accept": "application/json"}).
        body: Raw string body. Mutually exclusive with json_body / form_body.
        json_body: Dict/list — will be JSON-serialized and sent with
            Content-Type: application/json.
        form_body: Dict — will be URL-encoded and sent with
            Content-Type: application/x-www-form-urlencoded.
        timeout: Seconds before giving up. Default 30.
        follow_redirects: Whether to follow 3xx redirects. Default True.

    Returns:
        A formatted string with: status line, response headers (subset),
        and the body. If the body parses as JSON, the parsed structure is
        included too. Bodies >2 MiB are truncated with a note.
    """
    method = method.upper()
    if json_body is not None and (body is not None or form_body is not None):
        return "http_request error: json_body is mutually exclusive with body/form_body"
    if body is not None and form_body is not None:
        return "http_request error: body and form_body are mutually exclusive"

    req_headers = {"User-Agent": _DDG_UA, "Accept-Language": "en-US,en;q=0.9"}
    if headers:
        req_headers.update(headers)

    data: bytes | None = None
    if json_body is not None:
        try:
            data = json.dumps(json_body).encode("utf-8")
        except (TypeError, ValueError) as e:
            return f"http_request error: json_body not JSON-serializable: {e}"
        req_headers.setdefault("Content-Type", "application/json")
    elif form_body is not None:
        data = urllib.parse.urlencode(form_body).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
    elif body is not None:
        data = body.encode("utf-8")

    req = urllib.request.Request(url, data=data, method=method, headers=req_headers)

    try:
        # Build a no-op opener that does (or doesn't) follow redirects, so
        # users can opt out per-call (some APIs hand back signed redirect
        # URLs that break if urllib mangles them).
        if follow_redirects:
            opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler())
        else:
            opener = urllib.request.build_opener(NoRedirectHandler())
        with opener.open(req, timeout=timeout) as r:
            status = r.status
            resp_headers = dict(r.headers.items())
            raw = r.read(_HTTP_MAX_BYTES + 1)
            truncated = len(raw) > _HTTP_MAX_BYTES
            if truncated:
                raw = raw[:_HTTP_MAX_BYTES]
            charset = r.headers.get_content_charset() or "utf-8"
            try:
                text = raw.decode(charset, errors="replace")
            except LookupError:
                text = raw.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        # 4xx/5xx — return the body anyway if present, often the most useful part
        try:
            raw = e.read(_HTTP_MAX_BYTES)
            charset = e.headers.get_content_charset() if e.headers else None
            text = raw.decode(charset or "utf-8", errors="replace")
        except Exception:
            text = ""
        return _format_http_response(
            url, method, e.code, dict(e.headers.items()) if e.headers else {},
            text, truncated=False, error=f"HTTP {e.code} {e.reason}",
        )
    except Exception as e:
        return f"http_request error: {type(e).__name__}: {e}"

    return _format_http_response(
        url, method, status, resp_headers, text, truncated=truncated,
    )


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Handler that converts 3xx responses into HTTPError so the caller sees
    the redirect status/Location instead of being silently followed."""

    def _http_error(self, req, fp, code, msg, headers):
        del fp  # base-class signature; only used to build the HTTPError below
        raise urllib.error.HTTPError(
            req.full_url, code, msg, headers, None,
        )

    http_error_301 = http_error_302 = http_error_303 = http_error_307 = http_error_308 = _http_error


def _format_http_response(url, method, status, resp_headers, text, truncated, error=None):
    # Show only headers that are usually useful; the full set is noisy.
    keep = ("content-type", "content-length", "location", "date", "server",
            "set-cookie", "x-ratelimit-remaining", "x-request-id", "etag",
            "cache-control", "x-powered-by")
    headers_subset = {k: v for k, v in resp_headers.items() if k.lower() in keep}

    head = f"{method} {url}\nStatus: {status}"
    if error:
        head += f"  ({error})"
    if truncated:
        head += f"\n(body truncated at {_HTTP_MAX_BYTES} bytes)"

    parts = [head, "\nHeaders:"]
    if headers_subset:
        for k, v in headers_subset.items():
            parts.append(f"  {k}: {v}")
    else:
        parts.append("  (none notable)")

    parsed_json = None
    ctype = resp_headers.get("Content-Type", "")
    if "application/json" in ctype.lower() or (text and text.lstrip().startswith(("{", "["))):
        try:
            parsed_json = json.loads(text)
            parts.append(f"\nBody (JSON, {len(text)} bytes):")
            parts.append(json.dumps(parsed_json, indent=2)[:4000])
        except (ValueError, RecursionError):
            parts.append(f"\nBody (text, {len(text)} bytes):")
            parts.append(text[:4000])
    else:
        parts.append(f"\nBody (text, {len(text)} bytes):")
        parts.append(text[:4000])

    return "\n".join(parts)


def download_to_file(url: str, path: str, timeout: float = 60.0) -> str:
    """Download a URL to a local file path. Returns the path and byte count.

    Use for: PDFs, images, zips, CSV exports, anything binary. The HTTP
    tool's response is text-only and capped at 2 MiB; this tool streams to
    disk with no cap. Parent directories are created if missing.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _DDG_UA})
        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        with urllib.request.urlopen(req, timeout=timeout) as r, open(path, "wb") as f:
            total = 0
            while True:
                chunk = r.read(64 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
        return f"Downloaded {url} -> {path} ({total} bytes)"
    except Exception as e:
        return f"download_to_file error: {type(e).__name__}: {e}"


def web_fetch(url: str, max_chars: int = 8000) -> str:
    """Download a URL and return its visible text (scripts/styles stripped,
    tags removed, whitespace collapsed). Use after web_search to actually
    read the content of a source the user asked about. Returns up to
    max_chars characters. Falls back to the browser tools if the result
    is empty/short — that usually means the page is JS-rendered."""
    try:
        html = _http_get(url, timeout=25)
    except Exception as e:
        return f"web_fetch failed: {e}"

    body = _SCRIPT_RE.sub(" ", html)
    body = _STYLE_RE.sub(" ", body)
    title_m = _TITLE_RE.search(body)
    title = _TAG_RE.sub(" ", title_m.group(1)).strip() if title_m else ""
    text = _TAG_RE.sub(" ", body)
    text = _WS_RE.sub(" ", text).strip()

    head = f"URL: {url}\nTitle: {title}\n\n"
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n... (truncated at {max_chars} chars)"
    return head + text


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
    {"type": "function", "function": {
        "name": "web_search", "description": (
            "Search the web and return up to N (title, url, snippet) results. "
            "USE THIS FIRST whenever the user asks to 'find N sources / papers / "
            "articles / pages on X' — it's one HTTP request and avoids the slow "
            "navigate → snapshot → type → click dance in the browser. Returns "
            "plain text, so you can read it directly. If the user wants depth, "
            "feed the returned URLs into web_fetch."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "The search query."},
            "n": {"type": "integer", "description": "Max results to return (1-20). Default 8."},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "web_fetch", "description": (
            "Download a URL and return its visible text (scripts/styles/tags "
            "stripped). Use after web_search to actually READ the content of a "
            "source the user asked about. Returns the page title + up to 8000 "
            "chars of body text. If the body is empty/short, the page is "
            "JS-rendered — fall back to browser_navigate to that URL."),
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "Absolute http(s) URL to fetch."},
            "max_chars": {"type": "integer", "description": "Max body chars to return (1-50000). Default 8000."},
        }, "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "http_request", "description": (
            "Make any HTTP request (GET, POST, PUT, PATCH, DELETE) with custom "
            "headers and a body. USE THIS for anything that isn't a plain read: "
            "calling REST APIs, POSTing a form, hitting a webhook, fetching a "
            "JSON resource with an Authorization header, GraphQL queries, etc. "
            "Returns the status line, a subset of response headers, and the "
            "body (auto-parsed if JSON, truncated at 4000 chars in the response). "
            "Bodies >2 MiB are truncated. The tool has NONE of the user's "
            "browser session cookies — for sites that need login (gmail, X, "
            "banking), drive the browser instead with browser_navigate."),
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "Absolute http(s) URL."},
            "method": {"type": "string", "description": "HTTP method. Default GET."},
            "headers": {"type": "object", "description": "Extra request headers as a JSON object.",
                        "additionalProperties": {"type": "string"}},
            "body": {"type": "string", "description": "Raw string body. Mutually exclusive with json_body and form_body."},
            "json_body": {"description": "Dict/list — JSON-serialized, Content-Type: application/json. Mutually exclusive with body and form_body."},
            "form_body": {"type": "object", "description": "Dict — URL-encoded, Content-Type: application/x-www-form-urlencoded. Mutually exclusive with body and json_body.",
                          "additionalProperties": {"type": "string"}},
            "timeout": {"type": "number", "description": "Seconds. Default 30."},
            "follow_redirects": {"type": "boolean", "description": "Follow 3xx redirects. Default true."},
        }, "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "download_to_file", "description": (
            "Download a URL to a local file path on disk. USE THIS for binary "
            "or large files: PDFs, images, zips, CSV exports, anything where "
            "http_request's 2 MiB text response is the wrong shape. Streams to "
            "disk with no size cap. Creates parent directories if they don't "
            "exist. Returns the destination path and byte count."),
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "Absolute http(s) URL to download."},
            "path": {"type": "string", "description": "Destination file path on this PC (absolute, or relative to CWD)."},
            "timeout": {"type": "number", "description": "Seconds. Default 60."},
        }, "required": ["url", "path"]}}},
]

LOCAL_DISPATCH = {
    "open_url": open_url,
    "open_app": open_app,
    "create_github_repo": create_github_repo,
    "list_directory": list_directory,
    "web_search": web_search,
    "web_fetch": web_fetch,
    "http_request": http_request,
    "download_to_file": download_to_file,
}


def _print_server_hint(name: str) -> None:
    """Print a one-liner telling the user how to recover a failed MCP server."""
    hints = {
        "opera": (
            "     Opera isn't running with --remote-debugging-port=9222. "
            "Launch Opera normally (the Start Menu shortcut has the flag)\n"
            "     or run:  opera.exe --remote-debugging-port=9222"
        ),
        "github": (
            "     The GitHub MCP needs a GITHUB_PERSONAL_ACCESS_TOKEN env var. "
            "Get one at https://github.com/settings/tokens"
        ),
    }
    hint = hints.get(name)
    if hint:
        for line in hint.splitlines():
            print(line)


# ---------------------------------------------------------------------------
# Opera CDP auto-launch
#
# The `opera_cdp_mcp.py` server assumes port 9222 is already open. If the user
# launched Opera without --remote-debugging-port=9222 (e.g. from a pinned
# taskbar icon, or by double-clicking the .exe), the server's first HTTP
# probe fails, stdio closes, and the agent silently drops the browser tools.
#
# To keep the agent self-driving, before starting the opera MCP subprocess we
# probe 9222 and, if nothing answers, launch Opera ourselves with the flag.
# The launched process is detached (DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP)
# so it survives the agent exiting.
# ---------------------------------------------------------------------------

# Discover path. Confirmed via `Get-CimInstance Win32_Process` on this box.
_OPERA_EXE_FALLBACK = r"C:\Users\2024\AppData\Local\Programs\Opera\opera.exe"

# Where to put the isolated profile for the auto-launched Opera. Using a
# separate --user-data-dir is the key trick: when the user's real Opera is
# already running (without --remote-debugging-port), launching opera.exe with
# its same profile would relay to that instance and 9222 would never bind.
# Pointing at a separate dir spawns a fully independent browser that the
# MCP server can drive, leaving the user's existing session untouched.
_OPERA_ISOLATED_PROFILE = r"C:\Users\2024\AppData\Local\Programs\Opera\cdp-profile"

# Windows process-creation flags. From WinBase.h:
#   DETACHED_PROCESS        = 0x00000008  (no console inherited)
#   CREATE_NEW_PROCESS_GROUP = 0x00000200  (new process group, no Ctrl-C)
_DETACHED_PROCESS = 0x00000008
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_LAUNCH_TIMEOUT_S = 25.0
_PROBE_INTERVAL_S = 0.5
_CDP_PROBE_TIMEOUT_S = 0.5


def _cdp_port_open(http_endpoint: str) -> bool:
    """True if http_endpoint/json/version returns 200 within ~0.5s.

    Connection-refused on Windows can take 2s by default; we override the
    default timeout so the helper doesn't waste its 25s budget on probes.
    """
    try:
        req = urllib.request.Request(f"{http_endpoint}/json/version")
        with urllib.request.urlopen(req, timeout=_CDP_PROBE_TIMEOUT_S) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def _resolve_opera_launch() -> tuple[str, list[str]]:
    """Return (executable_path, extra_argv) for launching Opera with CDP.

    Always uses --user-data-dir pointing at a SEPARATE profile directory,
    so the launched browser is independent of any pre-existing Opera
    instance. Without this, Chrome would relay the new process to the
    existing browser (which doesn't have the CDP flag bound) and 9222
    would never come up.
    """
    if os.path.isfile(_OPERA_EXE_FALLBACK):
        os.makedirs(_OPERA_ISOLATED_PROFILE, exist_ok=True)
        return _OPERA_EXE_FALLBACK, [
            "--remote-debugging-port=9222",
            f"--user-data-dir={_OPERA_ISOLATED_PROFILE}",
        ]

    raise FileNotFoundError(
        f"Couldn't find opera.exe at '{_OPERA_EXE_FALLBACK}'. "
        f"Install Opera or fix the path."
    )


def _ensure_opera_cdp_running() -> None:
    """If port 9222 isn't open, launch an isolated Opera with CDP enabled.

    No-op when the port is already responding. Otherwise spawns Opera with
    --remote-debugging-port=9222 AND --user-data-dir pointing at a separate
    profile directory, so the new browser is fully independent of any
    pre-existing Opera. Polls the port until it answers or the timeout
    elapses.

    Raises on launch failure or timeout — the caller (MCPManager.connect_all)
    catches and surfaces the existing SKIPPED message.
    """
    http_endpoint = os.environ.get("OPERA_CDP_HTTP", "http://localhost:9222")
    if _cdp_port_open(http_endpoint):
        return  # already running; nothing to do

    exe, extra_args = _resolve_opera_launch()
    cmd = [exe] + extra_args
    print(f"  [auto-launch] port 9222 closed — launching isolated Opera with CDP")
    try:
        subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            creationflags=_DETACHED_PROCESS | _CREATE_NEW_PROCESS_GROUP,
        )
    except Exception as e:
        raise RuntimeError(
            f"failed to launch Opera at {exe}: {type(e).__name__}: {e}"
        ) from e

    # Poll the port until it answers. A fresh isolated Opera profile takes
    # ~5–8s to initialize on first run; 25s leaves headroom.
    t0 = time.time()
    while time.time() - t0 < _LAUNCH_TIMEOUT_S:
        if _cdp_port_open(http_endpoint):
            print(f"  [auto-launch] port 9222 up after {time.time() - t0:.1f}s")
            return
        time.sleep(_PROBE_INTERVAL_S)

    raise TimeoutError(
        f"launched Opera but {http_endpoint} didn't respond within "
        f"{_LAUNCH_TIMEOUT_S:.0f}s"
    )


class MCPManager:
    """Connects to configured MCP servers and routes tool calls to them."""

    def __init__(self):
        self.stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}   # tool_name -> session
        self.tool_specs: list[dict] = []

    async def connect_all(self):
        for server in MCP_SERVERS:
            try:
                # The Opera MCP server expects port 9222 to already be open.
                # If the user launched Opera without the flag, auto-launch
                # a second instance with the flag so the agent recovers on
                # its own instead of dropping the browser tools silently.
                if server["name"] == "opera":
                    _ensure_opera_cdp_running()

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
            except Exception as e:
                # Don't kill the whole agent over one bad MCP server. The
                # remaining tools (web_search, http_request, etc.) still work;
                # only the failed server's tools are missing.
                print(f"  SKIPPED {server['name']}: {type(e).__name__}: {e}")
                print(f"     -> {server['name']} tools won't be available this session")
                _print_server_hint(server["name"])

    async def call(self, name: str, args: dict) -> str:
        session = self.sessions.get(name)
        if session is None:
            return f"Unknown MCP tool: {name}"
        result = await session.call_tool(name, args)
        parts = [c.text for c in result.content if hasattr(c, "text")]
        return "\n".join(parts) if parts else str(result.content)

    async def close(self):
        await self.stack.aclose()


# Imperative verbs/phrases that map to specific tools. If a user message
# contains one of these AND the model returns text without calling any
# tool, qwen2.5:7b has been observed to "describe" doing the action
# instead of doing it — the hallucination guard in run_agent() catches
# that and re-prompts. False positives just cost one extra round-trip.
_ACTION_VERBS = (
    "open ", "open up ", "open the ", "navigate ", "go to ", "go ahead and ",
    "click ", "type ", "fetch ", "search ", "look up ", "find ", "find me ",
    "download ", "load ", "check ", "play ", "start a match", "start a game",
    "sign in", "log in", "log into", "buy ", "post ", "send ", "submit ",
    "create ", "make ", "run ", "launch ", "execute ", "fill out ", "scroll ",
    "refresh ", "close ", "switch ", "select ", "pick ", "choose ", "book ",
    "order ", "schedule ", "set up ", "setup ", "install ", "update ",
)


def _looks_like_action_request(user_input: str) -> bool:
    """Return True if the user message plausibly asks for a tool action.

    Used by the hallucination guard in run_agent() to detect when the model
    described an action without calling the matching tool. Whitespace-
    normalized, case-insensitive substring match.
    """
    if not user_input:
        return False
    s = " " + user_input.lower().strip() + " "
    return any(verb in s for verb in _ACTION_VERBS)


async def run_agent():
    mcp = MCPManager()
    print("Connecting to MCP servers...")
    if MCP_SERVERS:
        await mcp.connect_all()
    else:
        print("  (none configured)")

    all_tool_specs = LOCAL_TOOL_SPECS + mcp.tool_specs
    mcp_tool_names = {spec["function"]["name"] for spec in mcp.tool_specs}

    if mcp.tool_specs:
        print(f"  MCP tools ({len(mcp.tool_specs)}): {', '.join(sorted(mcp_tool_names))}")
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
            "  browser_open_tabs(urls) open multiple URLs at once, EACH in its "
            "OWN new tab. USE THIS whenever the user says 'open N tabs' or "
            "'open each of these' — don't loop browser_navigate (it reuses "
            "the active tab) and don't loop browser_tabs(action='open') "
            "(the model is unreliable at planning N separate calls).\n"
            "\n"
            "HARD RULE — NO NARRATION WITHOUT A TOOL CALL:\n"
            "Never describe a tool action without having actually called the "
            "tool in this same turn. If the user says 'open 3 tabs,' you MUST "
            "call browser_open_tabs(urls=[...]) — do NOT write text claiming "
            "you opened them. If you find yourself writing 'I opened...', "
            "'I've fetched...', 'I clicked...' and you have NOT yet seen a "
            "tool result confirming that action, STOP and call the tool "
            "instead. A previous version of this agent would happily describe "
            "opening tabs after only doing a web_search, and that was a bug.\n"
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
            "RESEARCH / DISCOVERY (web lookups, finding sources, fact-checking):\n"
            "When the user asks to find, look up, compare, or read things on\n"
            "the web (news sources, research papers, docs, prices, etc), use\n"
            "the lightweight text tools first, NOT the full browser dance:\n"
            "  web_search(query, n=8)    one HTTP call, returns N (title, url,\n"
            "                            snippet) tuples. Use this first.\n"
            "  web_fetch(url, max_chars) downloads a URL and returns the page\n"
            "                            title + up to 8000 chars of body text\n"
            "                            (scripts/styles stripped). Use this to\n"
            "                            actually read the content of a source.\n"
            "  http_request(url, method, headers, json_body/form_body/body)\n"
            "                            any HTTP method with custom headers\n"
            "                            and a body. Use for: REST APIs, POSTing\n"
            "                            forms, fetching JSON with auth headers,\n"
            "                            GraphQL queries, webhooks, anything that\n"
            "                            needs a non-GET or Authorization header.\n"
            "                            Returns status, headers, and body (auto-\n"
            "                            parsed if JSON, capped at 2 MiB).\n"
            "  download_to_file(url, path) streams a URL straight to a local\n"
            "                            file. Use for binary/large downloads\n"
            "                            (PDFs, images, zips, CSV exports) where\n"
            "                            http_request's text response is wrong.\n"
            "TYPICAL RESEARCH FLOW (e.g. 'find 5 news sources on topic X'):\n"
            "  1. web_search('X news', n=10)            -> pick 5 distinct outlets\n"
            "  2. web_fetch(url1)  for each of the 5   -> summarize each\n"
            "  3. Reply with: source name, URL, 2-3 sentence summary per item.\n"
            "TYPICAL API FLOW (e.g. 'check the weather for zip 94110'):\n"
            "  1. http_request('https://api.example.com/weather?zip=94110')  -> parse JSON\n"
            "  2. Reply with the relevant fields.\n"
            "FALLBACK RULES:\n"
            "  - If web_fetch returns <200 chars of body, the page is JS-\n"
            "    rendered; fall back to browser_navigate(url) + browser_snapshot\n"
            "    to read it the visual way.\n"
            "  - If web_search returns '(no results returned)', try rephrasing\n"
            "    the query (fewer words, drop quotes, add 'news' or 'paper').\n"
            "  - For sources that need login (gmail, banking, etc), NEVER use\n"
            "    web_search/web_fetch/http_request — they don't carry the user's\n"
            "    browser session. Use the browser tools, and stop at any\n"
            "    login/CAPTCHA wall and tell the user.\n"
            "  - For research papers, prefer queries like 'X arxiv' or 'X site:arxiv.org'.\n"
            "  - Cite the URL alongside every claim you make from a source.\n"
            "\n"
            "Keep replies short. If a tool result is an error, report the error "
            "honestly instead of claiming the action succeeded. NEVER claim a "
            "tool action succeeded unless a tool result told you it succeeded "
            "in this same turn — if you didn't see the result, the action did "
            "not happen."
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
            # uses ~9 tool calls on its own. A "find 5 sources and read each
            # one" research task is web_search + 5x web_fetch + reply = ~7,
            # so 40 leaves plenty of headroom for rephrases and fallbacks.
            MAX_STEPS = 40
            # Hallucination guard: if the user message contains an action verb
            # but the model returns text WITHOUT calling any tool, qwen2.5:7b
            # has been observed to "describe" doing the thing instead of
            # doing it. Track which tools were called this user turn and
            # re-prompt up to 2 times if the model wrote a description with
            # no tool calls behind it.
            tools_called_this_turn: set[str] = set()
            reprompts_used = 0
            MAX_REPROMPTS = 2
            for _ in range(MAX_STEPS):
                response = await client.chat(model=MODEL, messages=messages, tools=all_tool_specs)
                msg = response["message"]
                messages.append(msg)

                tool_calls = msg.get("tool_calls")
                if not tool_calls:
                    text = msg.get("content", "") or ""
                    if not tools_called_this_turn and _looks_like_action_request(user_input) \
                            and reprompts_used < MAX_REPROMPTS:
                        reprompts_used += 1
                        nudge = (
                            "You described doing this in your previous reply "
                            "without actually calling any tools. Call the required "
                            "tool now (e.g. browser_open_tabs, browser_navigate, "
                            "web_search, web_fetch) — do not just write what you "
                            "would have done. Reply only with the tool call."
                        )
                        print(f"  [hallucination guard: re-prompting ({reprompts_used}/{MAX_REPROMPTS})]")
                        messages.append({"role": "user", "content": nudge})
                        continue
                    print(text if text else "(no response)")
                    break

                for call in tool_calls:
                    fn_name = call["function"]["name"]
                    raw_args = call["function"]["arguments"]
                    # Ollama 0.6.x returns a dict; some versions / models
                    # return None, "", or a JSON string. Normalize defensively
                    # so a zero-arg tool (browser_snapshot, browser_tabs list)
                    # never crashes the agent loop on json.loads(None).
                    if isinstance(raw_args, dict):
                        args = raw_args
                    elif not raw_args:
                        args = {}
                    else:
                        try:
                            args = json.loads(raw_args)
                            if not isinstance(args, dict):
                                args = {"value": args}
                        except (TypeError, ValueError):
                            args = {}

                    print(f"  [running {fn_name}({args})]")
                    tools_called_this_turn.add(fn_name)
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