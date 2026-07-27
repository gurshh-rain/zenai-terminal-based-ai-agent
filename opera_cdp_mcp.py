"""
opera_cdp_mcp.py — Lightweight MCP server that drives your already-running
Opera over the Chrome DevTools Protocol (CDP).

Why this exists: @playwright/mcp hangs against Opera 133 (Chrome 149) on
`Page.addScriptToEvaluateOnNewDocument` / page-session handshake, so it can't
attach to a running browser. This server skips Playwright's high-level
abstractions and talks CDP directly over websockets, with the bare minimum
protocol dance needed to drive a page.

Tools exposed (MCP):
    browser_navigate(url)
    browser_snapshot()             — accessibility tree, with clickable CSS
    browser_click(selector)        — click a CSS selector
    browser_type(selector, text)   — type into a CSS-selected input
    browser_press_key(key)         — Escape, Enter, ArrowDown, etc.
    browser_tabs(action, ...)      — list / select / open / close tabs
    browser_evaluate(expression)   — run a JS expression, return value

This is started as a child process by agent.py and speaks MCP over stdio.
"""

import asyncio
import json
import os
import sys
from typing import Any

import websockets
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool


CDP_HTTP = os.environ.get("OPERA_CDP_HTTP", "http://localhost:9222")
DEFAULT_VIEWPORT = {"width": 1280, "height": 900}


# ---------------------------------------------------------------------------
# CDP client — one per attached page target
# ---------------------------------------------------------------------------

class CDPSession:
    """A logical CDP session bound to a (websocket, sessionId) pair.

    The actual websocket is shared across all sessions — only ONE coroutine
    may read from a websocket at a time, so we use a single shared
    reader (CDPReader) that demuxes messages to the right session.
    """

    def __init__(self, reader: "CDPReader", session_id: str | None = None):
        self._reader = reader
        self.session_id = session_id
        self._id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._closed = False

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def deliver(self, msg: dict):
        """Called by the reader for each message addressed to this session."""
        if self._closed:
            return
        if "id" in msg:
            fut = self._pending.pop(msg["id"], None)
            if fut and not fut.done():
                if "error" in msg:
                    fut.set_exception(CDPError(msg["error"]))
                else:
                    fut.set_result(msg.get("result"))
        # events are ignored for now; we only round-trip requests

    async def send(self, method: str, params: dict | None = None, timeout: float = 30.0) -> dict:
        if self._closed:
            raise CDPError("session closed")
        mid = self._next_id()
        payload: dict[str, Any] = {"id": mid, "method": method}
        if params:
            payload["params"] = params
        if self.session_id:
            payload["sessionId"] = self.session_id
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[mid] = fut
        await self._reader.send_raw(payload)
        return await asyncio.wait_for(fut, timeout=timeout)

    async def close(self):
        self._closed = True
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(CDPError("session closed"))
        self._pending.clear()


class CDPReader:
    """Owns the websocket and demuxes incoming messages to CDPSessions.

    Browser-level messages have no `sessionId`; per-target messages do.
    """

    def __init__(self, ws: websockets.WebSocketClientProtocol):
        self.ws = ws
        self._sessions_by_id: dict[str, CDPSession] = {}
        self._browser_session: CDPSession | None = None
        self._closed = False

    def register_browser(self, sess: CDPSession):
        self._browser_session = sess

    def register_session(self, session_id: str, sess: CDPSession):
        self._sessions_by_id[session_id] = sess

    def unregister_session(self, session_id: str):
        self._sessions_by_id.pop(session_id, None)

    async def send_raw(self, payload: dict):
        await self.ws.send(json.dumps(payload))

    async def run(self):
        """Read messages forever; dispatch to the right session."""
        try:
            async for raw in self.ws:
                if self._closed:
                    return
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                sid = msg.get("sessionId")
                if sid and sid in self._sessions_by_id:
                    self._sessions_by_id[sid].deliver(msg)
                elif self._browser_session is not None:
                    self._browser_session.deliver(msg)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._closed = True
    

class CDPError(Exception):
    def __init__(self, info):
        if isinstance(info, dict):
            super().__init__(info.get("message", json.dumps(info)))
            self.info = info
        else:
            super().__init__(str(info))
            self.info = {"message": str(info)}


# ---------------------------------------------------------------------------
# Browser — owns the browser-level websocket and a list of attached page sessions
# ---------------------------------------------------------------------------

class OperaBrowser:
    """Connects to Opera over CDP and tracks one active page session."""

    def __init__(self, http_endpoint: str = CDP_HTTP):
        self.http_endpoint = http_endpoint
        self.ws = None  # raw websocket
        self.reader: CDPReader | None = None
        self._browser_sess: CDPSession | None = None
        self.sessions: dict[str, CDPSession] = {}  # targetId -> CDPSession
        self.active_target: str | None = None

    async def connect(self):
        import urllib.request
        with urllib.request.urlopen(f"{self.http_endpoint}/json/version", timeout=5) as r:
            info = json.loads(r.read())
        ws_url = info["webSocketDebuggerUrl"]
        self.ws = await websockets.connect(ws_url, max_size=64 * 1024 * 1024)
        self.reader = CDPReader(self.ws)
        # Set up the browser-level session
        self._browser_sess = CDPSession(self.reader, session_id=None)
        self.reader.register_browser(self._browser_sess)
        # Start the demuxer loop
        asyncio.create_task(self.reader.run())
        # Enable Target domain
        await self._browser_sess.send("Target.setDiscoverTargets", {"discover": True})

    async def list_targets(self) -> list[dict]:
        r = await self._browser_sess.send("Target.getTargets")
        return r.get("targetInfos", [])

    async def attach_to(self, target_id: str) -> CDPSession:
        if target_id in self.sessions:
            self.active_target = target_id
            return self.sessions[target_id]
        r = await self._browser_sess.send("Target.attachToTarget", {
            "targetId": target_id,
            "flatten": True,
        })
        session_id = r["sessionId"]
        sess = CDPSession(self.reader, session_id=session_id)
        self.reader.register_session(session_id, sess)
        self.sessions[target_id] = sess
        self.active_target = target_id
        # Enable the domains we use
        await sess.send("Page.enable")
        await sess.send("Runtime.enable")
        await sess.send("Accessibility.enable")
        return sess

    async def new_page(self, url: str = "about:blank") -> tuple[str, CDPSession]:
        r = await self._browser_sess.send("Target.createTarget", {"url": url})
        target_id = r["targetId"]
        sess = await self.attach_to(target_id)
        if url != "about:blank":
            await sess.send("Page.navigate", {"url": url})
            await self._wait_for_load(sess)
        return target_id, sess

    async def _wait_for_load(self, sess: CDPSession, timeout: float = 30.0):
        import time
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                r = await sess.send("Runtime.evaluate", {
                    "expression": "document.readyState",
                    "returnByValue": True,
                })
                if r.get("result", {}).get("value") in ("complete", "interactive"):
                    await asyncio.sleep(0.3)
                    return
            except Exception:
                pass
            await asyncio.sleep(0.2)
        raise CDPError("page load timed out")

    async def close_page(self, target_id: str):
        sess = self.sessions.pop(target_id, None)
        if sess:
            if sess.session_id:
                self.reader.unregister_session(sess.session_id)
            sess.close()
        try:
            await self._browser_sess.send("Target.closeTarget", {"targetId": target_id})
        except Exception:
            pass
        if self.active_target == target_id:
            self.active_target = None

    async def shutdown(self):
        for s in self.sessions.values():
            s.close()
        self.sessions.clear()
        if self._browser_sess:
            self._browser_sess.close()
        if self.reader:
            self.reader._closed = True
        if self.ws:
            await self.ws.close()


# ---------------------------------------------------------------------------
# Snapshot / click helpers
# ---------------------------------------------------------------------------

INTERACTIVE_ROLES = {
    "link", "button", "textbox", "combobox", "checkbox", "radio",
    "menuitem", "menuitemcheckbox", "menuitemradio", "tab", "switch",
    "searchbox", "spinbutton", "slider", "option", "treeitem",
}


async def page_snapshot(sess: CDPSession) -> dict:
    """Build a snapshot of the current page: title, URL, and a list of
    interactive nodes with stable CSS selectors the model can click.

    We walk the live DOM via Runtime.evaluate and synthesize a small
    accessibility-style listing, because some sites (chess.com in
    particular) don't expose a useful tree through the CDP
    `Accessibility.getFullAXTree` domain.
    """
    r = await sess.send("Runtime.evaluate", {
        "expression": "({title: document.title, url: location.href})",
        "returnByValue": True,
    })
    title = r.get("result", {}).get("value", {}).get("title", "")
    url = r.get("result", {}).get("value", {}).get("url", "")

    # Walk the live DOM. For each interactive-ish element, generate a
    # CSS selector and a human label.
    walk_js = r"""
    (() => {
      const SEL = (el) => {
        if (!(el instanceof Element)) return null;
        if (el.id && document.querySelectorAll('#' + CSS.escape(el.id)).length === 1) {
          return '#' + CSS.escape(el.id);
        }
        const parts = [];
        let cur = el;
        let depth = 0;
        while (cur && cur.nodeType === 1 && depth < 7) {
          let part = cur.tagName.toLowerCase();
          const role = cur.getAttribute('role');
          if (role) part += '[role="' + role + '"]';
          const parent = cur.parentElement;
          if (parent) {
            const sameTag = Array.from(parent.children).filter(c => c.tagName === cur.tagName);
            if (sameTag.length > 1) {
              const idx = sameTag.indexOf(cur) + 1;
              part += ':nth-of-type(' + idx + ')';
            }
          }
          parts.unshift(part);
          const candidate = parts.join(' > ');
          if (document.querySelectorAll(candidate).length === 1) return candidate;
          cur = parent;
          depth++;
        }
        return parts.join(' > ');
      };
      const LABEL = (el) => {
        if (el.getAttribute('aria-label')) return el.getAttribute('aria-label').trim();
        const al = el.querySelector('[aria-label]');
        if (al) return al.getAttribute('aria-label').trim();
        // text of immediate non-empty children
        const txt = (el.innerText || el.textContent || '').trim();
        return txt.slice(0, 100);
      };
      const out = [];
      const candidates = document.querySelectorAll(
        'a, button, input, textarea, select, [role="button"], [role="link"], [role="menuitem"], [role="tab"], [role="checkbox"], [role="radio"], [role="switch"], [role="combobox"], [contenteditable="true"]'
      );
      for (const el of candidates) {
        // skip hidden
        const r = el.getBoundingClientRect();
        if (r.width === 0 && r.height === 0) continue;
        const cs = getComputedStyle(el);
        if (cs.visibility === 'hidden' || cs.display === 'none' || +cs.opacity === 0) continue;
        const sel = SEL(el);
        if (!sel) continue;
        const tag = el.tagName.toLowerCase();
        const type = el.getAttribute('type') || '';
        const role = el.getAttribute('role') || '';
        let kind = tag;
        if (tag === 'input') kind = 'input[' + (type || 'text') + ']';
        else if (role) kind = role;
        out.push({
          kind,
          name: LABEL(el),
          selector: sel,
          href: el.getAttribute('href') || '',
          value: el.value || '',
        });
        if (out.length > 200) break;
      }
      return out;
    })()
    """
    r2 = await sess.send("Runtime.evaluate", {
        "expression": walk_js,
        "returnByValue": True,
    })
    nodes = (r2.get("result") or {}).get("value") or []

    lines: list[str] = []
    for n in nodes:
        label = n.get("name") or n.get("value") or n.get("href") or ""
        lines.append(f'{n["kind"]}: "{label[:80]}" → {n["selector"]}')

    return {
        "title": title,
        "url": url,
        "interactive_count": len(nodes),
        "nodes": nodes,
        "text": "\n".join(lines) if lines else "(no interactive elements found)",
    }


_SELECTOR_JS = r"""
function () {
  function uniqueSelector(el) {
    if (!(el instanceof Element)) return null;
    // Prefer a stable id
    if (el.id && document.querySelectorAll('#' + CSS.escape(el.id)).length === 1) {
      return '#' + CSS.escape(el.id);
    }
    // Build a path of tag + attributes + nth-child
    const parts = [];
    let cur = el;
    let depth = 0;
    while (cur && cur.nodeType === 1 && depth < 6) {
      let part = cur.tagName.toLowerCase();
      const role = cur.getAttribute('role');
      if (role) part += '[role="' + role + '"]';
      const name = cur.getAttribute('aria-label') || cur.getAttribute('title') || (cur.textContent || '').trim().slice(0, 40);
      // use :has-text via :is() not supported; rely on tag only unless needed
      // Add nth-of-type if not unique among siblings
      const parent = cur.parentElement;
      if (parent) {
        const sameTag = Array.from(parent.children).filter(c => c.tagName === cur.tagName);
        if (sameTag.length > 1) {
          const idx = sameTag.indexOf(cur) + 1;
          part += ':nth-of-type(' + idx + ')';
        }
      }
      parts.unshift(part);
      // check uniqueness from the top
      const candidate = parts.join(' > ');
      if (document.querySelectorAll(candidate).length === 1) {
        return candidate;
      }
      cur = parent;
      depth++;
    }
    return parts.join(' > ');
  }
  return uniqueSelector(this);
}
"""

# Wrap JS into a function string the way CDP expects
_selector_fn = "function() { " + _SELECTOR_JS[len("function () {"):] if _SELECTOR_JS.startswith("function ()") else _SELECTOR_JS


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

server = Server("opera-cdp")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(name="browser_navigate", description=(
            "Open a URL in a new tab in the user's running Opera browser. "
            "Returns the page title, URL, and a snapshot of interactive elements. "
            "For any site you don't already have a tab for, prefer this over guessing paths."
        ), inputSchema={
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        }),
        Tool(name="browser_snapshot", description=(
            "Re-read the current tab and return its title, URL, and the list of "
            "interactive elements (links, buttons, inputs) with stable CSS selectors "
            "you can pass to browser_click / browser_type."
        ), inputSchema={"type": "object", "properties": {}}),
        Tool(name="browser_click", description=(
            "Click an element on the current tab. The `selector` is a CSS selector "
            "you got from a previous browser_snapshot output."
        ), inputSchema={
            "type": "object",
            "properties": {"selector": {"type": "string"}},
            "required": ["selector"],
        }),
        Tool(name="browser_type", description=(
            "Type text into an input on the current tab. The `selector` should "
            "target the input element."
        ), inputSchema={
            "type": "object",
            "properties": {
                "selector": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["selector", "text"],
        }),
        Tool(name="browser_press_key", description=(
            "Press a single key on the current tab. Common keys: Enter, Escape, "
            "Tab, ArrowDown, ArrowUp, /."
        ), inputSchema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        }),
        Tool(name="browser_tabs", description=(
            "Manage tabs. action=list shows all open tabs; action=open opens a "
            "new tab (and optionally navigates to url); action=select switches "
            "the active tab to the given targetId; action=close closes a tab."
        ), inputSchema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "open", "select", "close"]},
                "url": {"type": "string"},
                "targetId": {"type": "string"},
            },
            "required": ["action"],
        }),
        Tool(name="browser_evaluate", description=(
            "Run a JavaScript expression on the current tab and return the result. "
            "Use this for things the higher-level tools can't express."
        ), inputSchema={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        }),
    ]


def _format_snapshot(snap: dict) -> str:
    return (
        f"Title: {snap['title']}\n"
        f"URL: {snap['url']}\n"
        f"Interactive elements ({snap['interactive_count']}):\n"
        f"{snap['text']}"
    )


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    global browser
    try:
        if name == "browser_navigate":
            url = arguments["url"]
            # Reuse active tab if one exists, otherwise open a new one.
            if browser.active_target and browser.active_target in browser.sessions:
                sess = browser.sessions[browser.active_target]
                await sess.send("Page.navigate", {"url": url})
                await browser._wait_for_load(sess)
            else:
                target_id, sess = await browser.new_page(url)
                browser.active_target = target_id
            snap = await page_snapshot(sess)
            return [TextContent(type="text", text=_format_snapshot(snap))]

        if name == "browser_snapshot":
            if not browser.active_target:
                return [TextContent(type="text", text="(no active tab — call browser_navigate or browser_tabs action=open first)")]
            sess = browser.sessions[browser.active_target]
            snap = await page_snapshot(sess)
            return [TextContent(type="text", text=_format_snapshot(snap))]

        if name == "browser_click":
            selector = arguments["selector"]
            if not browser.active_target:
                return [TextContent(type="text", text="(no active tab)")]
            sess = browser.sessions[browser.active_target]
            # scroll into view, then click
            await sess.send("Runtime.evaluate", {
                "expression": (
                    "(()=>{const e=document.querySelector(" + json.dumps(selector) + ");"
                    "if(!e)return false;e.scrollIntoView({block:'center'});return true;})()"
                ),
                "returnByValue": True,
            })
            # get box
            doc = await sess.send("Runtime.evaluate", {
                "expression": (
                    "(()=>{const e=document.querySelector(" + json.dumps(selector) + ");"
                    "if(!e)return null;const r=e.getBoundingClientRect();"
                    "return {x:r.x+r.width/2, y:r.y+r.height/2, w:window.innerWidth, h:window.innerHeight};})()"
                ),
                "returnByValue": True,
            })
            box = (doc.get("result") or {}).get("value")
            if not box:
                return [TextContent(type="text", text=f"(selector not found: {selector})")]
            # Use Input.dispatchMouseEvent for a real click
            await sess.send("Input.dispatchMouseEvent", {
                "type": "mousePressed", "x": box["x"], "y": box["y"],
                "button": "left", "clickCount": 1,
            })
            await sess.send("Input.dispatchMouseEvent", {
                "type": "mouseReleased", "x": box["x"], "y": box["y"],
                "button": "left", "clickCount": 1,
            })
            await asyncio.sleep(0.5)
            snap = await page_snapshot(sess)
            return [TextContent(type="text", text=f"clicked {selector}\n\n" + _format_snapshot(snap))]

        if name == "browser_type":
            selector = arguments["selector"]
            text = arguments["text"]
            if not browser.active_target:
                return [TextContent(type="text", text="(no active tab)")]
            sess = browser.sessions[browser.active_target]
            # focus
            await sess.send("Runtime.evaluate", {
                "expression": (
                    "(()=>{const e=document.querySelector(" + json.dumps(selector) + ");"
                    "if(e){e.focus();e.value='';}return !!e;})()"
                ),
                "returnByValue": True,
            })
            # type each char
            for ch in text:
                await sess.send("Input.dispatchKeyEvent", {
                    "type": "char", "text": ch,
                })
            await asyncio.sleep(0.2)
            return [TextContent(type="text", text=f"typed {len(text)} chars into {selector}")]

        if name == "browser_press_key":
            key = arguments["key"]
            if not browser.active_target:
                return [TextContent(type="text", text="(no active tab)")]
            sess = browser.sessions[browser.active_target]
            await sess.send("Input.dispatchKeyEvent", {
                "type": "keyDown", "key": key, "code": key, "windowsVirtualKeyCode": _vk(key),
            })
            await sess.send("Input.dispatchKeyEvent", {
                "type": "keyUp", "key": key, "code": key, "windowsVirtualKeyCode": _vk(key),
            })
            await asyncio.sleep(0.3)
            return [TextContent(type="text", text=f"pressed {key}")]

        if name == "browser_tabs":
            action = arguments.get("action")
            if action == "list":
                targets = await browser.list_targets()
                pages = [t for t in targets if t.get("type") == "page"]
                lines = [f'- {t["targetId"]}  {t.get("title","")[:50]}  {t.get("url","")[:80]}' for t in pages]
                return [TextContent(type="text", text="open tabs:\n" + "\n".join(lines) if lines else "(none)")]
            if action == "open":
                url = arguments.get("url", "about:blank")
                tid, _ = await browser.new_page(url)
                return [TextContent(type="text", text=f"opened tab {tid} → {url}")]
            if action == "select":
                tid = arguments["targetId"]
                await browser.attach_to(tid)
                sess = browser.sessions[tid]
                snap = await page_snapshot(sess)
                return [TextContent(type="text", text=f"selected {tid}\n\n" + _format_snapshot(snap))]
            if action == "close":
                tid = arguments.get("targetId") or browser.active_target
                if tid:
                    await browser.close_page(tid)
                return [TextContent(type="text", text=f"closed {tid}")]
            return [TextContent(type="text", text=f"unknown action: {action}")]

        if name == "browser_evaluate":
            expr = arguments["expression"]
            if not browser.active_target:
                return [TextContent(type="text", text="(no active tab)")]
            sess = browser.sessions[browser.active_target]
            r = await sess.send("Runtime.evaluate", {
                "expression": expr, "returnByValue": True, "awaitPromise": True,
            })
            res = r.get("result", {})
            if "value" in res:
                return [TextContent(type="text", text=json.dumps(res["value"], indent=2))]
            if res.get("type") == "undefined":
                return [TextContent(type="text", text="(undefined)")]
            return [TextContent(type="text", text=json.dumps(res, indent=2))]

        return [TextContent(type="text", text=f"unknown tool: {name}")]
    except Exception as e:
        return [TextContent(type="text", text=f"error: {e}")]


def _vk(key: str) -> int:
    table = {
        "Enter": 13, "Escape": 27, "Tab": 9, "Backspace": 8, "Delete": 46,
        "ArrowDown": 40, "ArrowUp": 38, "ArrowLeft": 37, "ArrowRight": 39,
        "Home": 36, "End": 35, "PageUp": 33, "PageDown": 34, " ": 32, "/": 191,
    }
    return table.get(key, 0)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

browser: OperaBrowser

async def main():
    global browser
    browser = OperaBrowser()
    await browser.connect()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
