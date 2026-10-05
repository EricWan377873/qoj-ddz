#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QOJ 斗地主：单局、九局比赛、本地 CPU 机器人和共享规则/模型。

在线：python client/qoj_cli.py
离线训练：python train.py；本地对战：python local.py
来源、许可、依赖和操作说明见唯一的 README.md；第三方许可见 LICENSE.txt。
DouZero: 718a5c920bf3361e34178a38f3b80458e176b351 (Apache-2.0).
AlphaDou: 13e740c08c3b653c2bef6ca345fc8fa6adc7d362 (GPL-3.0), Net2: Vincentzyx.
本版本将原模块合并为普通 Python 定义，保持网络协议、模型结构和权重格式。
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import getpass
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http.cookiejar import Cookie
from urllib.parse import parse_qs, urljoin, urlparse
from functools import lru_cache
import random
import numpy as np
import copy
from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
import tempfile
import multiprocessing as mp
import threading
import zipfile

# QOJ transport, card rules and terminal interface

ORIGIN = "https://qoj.ac"
LOBBY = "/games/doudizhu"
API = LOBBY + "/api"
COOKIE_NAME = "__Host-UOJSESSID"
RANKS = tuple("3456789XJQKA2SD")  # X=10，S=小王，D=大王
TYPE_ORDER = ("rocket", "bomb", "single", "pair", "trio", "straight", "pairs",
              "plane", "trio1", "trio2", "four2", "four22", "plane1", "plane2")
TYPE_NAMES = dict(zip(TYPE_ORDER, ("王炸", "炸弹", "单张", "对子", "三张", "顺子",
                                  "连对", "飞机", "三带一", "三带一对", "四带二",
                                  "四带两对", "飞机带单", "飞机带对")))


class ClientError(Exception):
    pass


class LoginError(ClientError):
    pass


class NetworkError(ClientError):
    """A request may have reached the server. Never automatically retry a move."""


class ApiError(ClientError):
    def __init__(self, message, payload=None, status=0, retry_after=0):
        super().__init__(message)
        self.payload = payload or {}
        self.status = status
        self.retry_after = retry_after


def clean(value) -> str:
    """Render remote text literally, with no terminal controls or bidi overrides."""
    return "".join(c if unicodedata.category(c) not in ("Cc", "Cf", "Cs") else " "
                   for c in str(value))


def parse_cookie(raw: str) -> str:
    raw = raw.strip()
    if COOKIE_NAME + "=" in raw:
        raw = raw.split(COOKIE_NAME + "=", 1)[1].split(";", 1)[0].strip()
    if not raw or not re.fullmatch(r"[A-Za-z0-9,._~%+/-]+", raw):
        raise ClientError("请输入 __Host-UOJSESSID 的值，或包含该字段的 Cookie 文本。")
    return raw


def rank(card: int) -> int:
    return card // 4 if card < 52 else card - 39


def cards_text(cards, grouped=False, desc=False) -> str:
    counts = Counter(rank(c) for c in cards or [])
    items = [RANKS[r] * counts[r] for r in sorted(counts, reverse=desc)]
    return (" " if grouped else "").join(items) or "∅"


def parse_ranks(text: str):
    """None means ordinary chat; [] is never returned for a nonempty play."""
    s = unicodedata.normalize("NFKC", text).strip().upper()
    s = s.replace("小王", "S").replace("大王", "D").replace("10", "X").replace("T", "X")
    s = re.sub(r"[\s,，]+", "", s)
    if not s or any(c not in RANKS for c in s):
        return None
    return [RANKS.index(c) for c in s]


def select_cards(hand, ranks):
    groups = {r: [] for r in range(15)}
    for card in sorted(hand or []):
        groups[rank(card)].append(card)
    needed = Counter(ranks)
    for r, n in needed.items():
        if len(groups[r]) < n:
            raise ClientError(f"手牌中没有 {n} 张 {RANKS[r]}。用 /say 文本 可强制聊天。")
    return sorted(c for r, n in needed.items() for c in groups[r][:n])


def classify(cards):
    """Match QOJ's rank/len/type and ambiguous plane semantics, not generic DDZ."""
    n = len(cards)
    if not n or len(set(cards)) != n or any(not 0 <= c < 54 for c in cards):
        return []
    c = [0] * 15
    for card in cards:
        c[rank(card)] += 1
    out = []

    def add(kind, r, length=1):
        out.append({"type": kind, "rank": r, "len": length})

    def of_count(k):
        return [r for r, count in enumerate(c) if count == k]

    def kickers_ok(rest):
        return not (rest[13] and rest[14]) and max(rest) < 4

    distinct = sum(v > 0 for v in c)
    if n == 2 and c[13] and c[14]:
        add("rocket", 14)
    if n == 1:
        add("single", rank(cards[0]))
    for size, kind in ((2, "pair"), (3, "trio"), (4, "bomb")):
        if n == size and of_count(size):
            add(kind, of_count(size)[0])
    if n == 4 and distinct == 2 and of_count(3):
        add("trio1", of_count(3)[0])
    if n == 5 and distinct == 2 and of_count(3) and of_count(2):
        add("trio2", of_count(3)[0])
    for kind, width, minimum in (("straight", 1, 5), ("pairs", 2, 3), ("plane", 3, 2)):
        ranks = [r for r in range(15) if c[r]]
        if (len(ranks) >= minimum and ranks[-1] <= 11
                and ranks[-1] - ranks[0] == len(ranks) - 1
                and all(c[r] == width for r in ranks)):
            add(kind, ranks[-1], len(ranks))
    for r in of_count(4):
        rest = c.copy()
        rest[r] = 0
        if n == 6 and kickers_ok(rest):
            add("four2", r)
        if n == 8 and rest.count(2) == 2 and sum(v > 0 for v in rest) == 2:
            add("four22", r)
    for unit, kind in ((4, "plane1"), (5, "plane2")):
        if n % unit or n // unit < 2:
            continue
        k = n // unit
        for start in range(13 - k):
            if any(c[r] != 3 for r in range(start, start + k)):
                continue
            rest = c.copy()
            rest[start:start + k] = [0] * k
            if kind == "plane1" and not kickers_ok(rest):
                continue
            if kind == "plane2" and (any(v not in (0, 2) for v in rest) or rest.count(2) != k):
                continue
            add(kind, start + k - 1, k)
    return sorted(out, key=lambda p: (TYPE_ORDER.index(p["type"]), -p["rank"]))


def beats(a, b):
    if not b:
        return True
    if b["type"] == "rocket":
        return False
    if a["type"] == "rocket":
        return True
    if a["type"] == "bomb":
        return b["type"] != "bomb" or a["rank"] > b["rank"]
    return a["type"] == b["type"] and a["len"] == b["len"] and a["rank"] > b["rank"]


def pattern_key(p):
    return f"{p['type']}:{p['rank']}:{p['len']}"


def describe(p):
    if not p:
        return ""
    return TYPE_NAMES.get(p["type"], p["type"]) + ("" if p["type"] == "rocket" else " " + RANKS[p["rank"]])


def hints(hand, last):
    """Use the same limited hint strategy as the official frontend (no auto-play)."""
    g = [[] for _ in range(15)]
    for card in sorted(hand):
        g[rank(card)].append(card)
    cand = []

    def broken(r, take):
        return int(bool(g[13] and g[14])) if r >= 13 else int(len(g[r]) > take)

    def kickers(exclude, k, unit):
        ranks = [r for r in range(15) if r not in exclude and len(g[r]) >= unit and (unit != 2 or r < 13)]
        ranks.sort(key=lambda r: (broken(r, unit), r))
        if unit == 2:
            return [c for r in ranks[:k] for c in g[r][:2]] if len(ranks) >= k else None
        out = [g[r][layer] for layer in range(3) for r in ranks if len(g[r]) > layer]
        return out[:k] if len(out) >= k else None

    if not last:
        for r, group in enumerate(g):
            if group and not (r >= 13 and g[13] and g[14]):
                cand.append(((int(len(group) == 4), r), group))
    else:
        t, top, length = last["type"], last["rank"], last["len"]
        width = {"single": 1, "pair": 2, "trio": 3, "trio1": 3, "trio2": 3}.get(t)
        if width:
            for r in range(top + 1, 15):
                if len(g[r]) < width:
                    continue
                body = g[r][:width]
                if t in ("trio1", "trio2"):
                    kick = kickers({r}, 1, 1 if t == "trio1" else 2)
                    if kick is None:
                        continue
                    body += kick
                cand.append(((0, broken(r, width), r), body))
        width = {"straight": 1, "pairs": 2, "plane": 3, "plane1": 3, "plane2": 3}.get(t)
        if width:
            for end in range(top + 1, 12):
                start = end - length + 1
                if start < 0 or any(len(g[r]) < width for r in range(start, end + 1)):
                    continue
                body = [c for r in range(start, end + 1) for c in g[r][:width]]
                if t in ("plane1", "plane2"):
                    kick = kickers(set(range(start, end + 1)), length, 1 if t == "plane1" else 2)
                    if kick is None:
                        continue
                    body += kick
                cand.append(((0, int(any(broken(r, width) for r in range(start, end + 1))), end), body))
        if t in ("four2", "four22"):
            for r in range(top + 1, 13):
                if len(g[r]) == 4:
                    kick = kickers({r}, 2, 1 if t == "four2" else 2)
                    if kick is not None:
                        cand.append(((0, 0, r), g[r] + kick))
        if t != "rocket":
            for r in range(13):
                if len(g[r]) == 4 and (t != "bomb" or r > top):
                    cand.append(((1, 0, r), g[r]))
    if g[13] and g[14] and (not last or last["type"] != "rocket"):
        cand.append(((2, 0, 14), [52, 53]))
    seen, result = set(), []
    for _, cards in sorted(cand, key=lambda item: (len(item[0]), *item[0])):
        key = tuple(sorted(cards))
        if key not in seen and any(beats(p, last) for p in classify(cards)):
            seen.add(key)
            result.append(list(key))
    return result


@dataclass
class LobbyInfo:
    token: str = ""
    username: str = ""
    rating: str = "—"
    stats: dict = field(default_factory=dict)
    game: int | None = None


def soup_of(html):
    from bs4 import BeautifulSoup
    return BeautifulSoup(html, "html.parser")


def parse_lobby(html):
    soup = soup_of(html)
    root = soup.select_one("#ddz-lobby")
    if root is None:
        if soup.select_one('input[type="password"]') or soup.find("a", href=re.compile(r"/login")):
            raise LoginError("Cookie 无效或已过期，请退出并重新输入 __Host-UOJSESSID。")
        raise ClientError("没有找到 #ddz-lobby；页面结构可能已变，请提供最新大厅 HTML。")
    stats = {}
    for node in soup.select(".card-body .text-muted.small"):
        label = node.get_text(strip=True)
        value = node.parent.select_one(".h4")
        if value and label in ("总积分", "对局", "胜率", "地主胜率"):
            stats[label] = value.get_text(strip=True)
    hero = soup.select_one(".ddz-lobby-hero")
    rating_node = hero.select_one("p strong") if hero else None
    history = soup.find("a", href=re.compile(r"/games/doudizhu/history\?user="))
    username = parse_qs(urlparse(history["href"]).query).get("user", [""])[0] if history else ""
    token = root.get("data-token", "")
    if not token:
        raise LoginError("大厅没有 CSRF token，请检查 Cookie。")
    return LobbyInfo(token, username, rating_node.get_text(strip=True) if rating_node else "—",
                     stats, int(root["data-game"]) if root.get("data-game") else None)


def parse_game(html):
    soup = soup_of(html)
    root, initial = soup.select_one("#ddz"), soup.select_one("#ddz-initial")
    if root is None or initial is None:
        if soup.select_one('input[type="password"]'):
            raise LoginError("登录已过期，请重新运行并输入 Cookie。")
        raise ClientError("没有找到 #ddz / #ddz-initial，请提供最新对局 HTML。")
    data = json.loads(initial.get_text())
    if root.get("data-logged-in") != "1" or data["state"].get("seat") is None:
        raise LoginError("该 Cookie 不是这局的参赛者，不能出牌。")
    if data["state"].get("match"):
        raise ClientError("账号正在记分比赛中；本版只处理单局，请先在网页完成比赛。")
    return root.get("data-token", ""), data


@dataclass
class Reply:
    status: int
    text: str
    headers: dict = field(default_factory=dict)
    url: str = ""


def is_challenge(reply):
    head = reply.text[:12000].lower()
    is_html = "html" in reply.headers.get("content-type", "").lower() or head.lstrip().startswith("<")
    return (reply.headers.get("cf-mitigated") == "challenge" or
            (is_html and any(x in head for x in ("<title>just a moment", "window._cf_chl_opt",
                                                 "/cdn-cgi/challenge-platform/", "cf-chl-widget"))))


class Transport:
    """Serialized HTTP; optional browser fetch keeps clearance in its own browser."""
    def __init__(self, cookie, mode="auto", channel="auto", timeout=12, notify=print):
        from curl_cffi import requests
        self.session = requests.Session(impersonate="chrome", timeout=timeout,
                                        headers={"Accept-Language": "zh-CN,zh;q=0.9"})
        self.session.cookies.jar.set_cookie(Cookie(
            0, COOKIE_NAME, cookie, None, False, "qoj.ac", False, False,
            "/", True, True, None, True, None, None, {"HttpOnly": None}, False))
        self.mode, self.channel, self.timeout = mode, channel, timeout
        self.notify = notify
        self.browser = self.context = self.page = self.pw = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qoj-http")
        self.lock = asyncio.Lock()
        self.browser_attempted = False

    @property
    def label(self):
        return "浏览器连接" if self.page else "HTTP 连接"

    async def _direct(self, method, path, data=None):
        def perform():
            url = ORIGIN + path
            for _ in range(4):
                r = self.session.request(method, url, data=data, allow_redirects=False,
                                         headers={"Referer": ORIGIN + LOBBY, "Origin": ORIGIN})
                if r.status_code in (301, 302, 303, 307, 308):
                    target = urljoin(url, r.headers.get("location", ""))
                    parsed = urlparse(target)
                    if parsed.scheme != "https" or parsed.netloc != "qoj.ac":
                        raise ClientError("QOJ 返回了非本站重定向，已停止该请求。")
                    if method != "GET":
                        raise LoginError("接口返回重定向，请检查登录 Cookie 后重新运行。")
                    url = target
                    continue
                return Reply(r.status_code, r.text, dict(r.headers), r.url)
            raise ClientError("页面重定向次数过多。")
        try:
            return await asyncio.get_running_loop().run_in_executor(self.executor, perform)
        except ClientError:
            raise
        except Exception:
            raise NetworkError("网络请求失败或超时；请检查连接。") from None

    async def enable_browser(self):
        if self.page and not self.page.is_closed():
            await self.verify_browser()
            return
        for obj in (self.context, self.browser):
            if obj:
                with contextlib.suppress(Exception):
                    await obj.close()
        self.page = self.context = self.browser = None
        self.browser_attempted = True
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ClientError("浏览器回退需要 pip install playwright，然后 python -m playwright install chromium。") from None
        self.notify("需要浏览器验证：请在弹出的窗口完成 Cloudflare 验证，窗口保持打开。")
        if self.pw is None:
            self.pw = await async_playwright().start()
        channels = (["msedge", "chrome", None] if os.name == "nt" else ["chrome", None]) if self.channel == "auto" else [None if self.channel == "chromium" else self.channel]
        for channel in channels:
            try:
                self.browser = await self.pw.chromium.launch(headless=False, channel=channel)
                break
            except Exception:
                pass
        else:
            raise ClientError("浏览器启动失败。运行 python -m playwright install chromium；Linux 需图形桌面及浏览器系统依赖。")
        self.context = await self.browser.new_context(locale="zh-CN")
        cookies = []
        for c in self.session.cookies.jar:
            if c.domain.lstrip(".") == "qoj.ac":
                cookies.append({"name": c.name, "value": c.value, "url": ORIGIN + "/", "secure": True})
        await self.context.add_cookies(cookies)
        # Prevent the real lobby/game JavaScript from issuing duplicate API polls or actions.
        await self.context.route(re.compile(r"https://qoj\.ac/js/games/doudizhu(?:-lobby|-rules)?\.js(?:\?.*)?$"),
                                 lambda route: route.abort())
        self.page = await self.context.new_page()
        await self.verify_browser()

    async def verify_browser(self):
        try:
            await self.page.goto(ORIGIN + LOBBY + "?locale=zh-cn", wait_until="domcontentloaded", timeout=30000)
        except Exception:
            pass  # A challenge may keep navigation pending; inspect the page below.
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            try:
                if self.page.is_closed():
                    break
                if await self.page.locator("#ddz-lobby").count():
                    self.notify("浏览器验证通过，已使用该窗口的会话继续连接。")
                    return
                if await self.page.locator('input[type="password"]').count():
                    raise LoginError("Cookie 无效或已过期，请重新运行输入 Cookie。")
            except LoginError:
                raise
            except Exception:
                pass
            await asyncio.sleep(0.6)
        raise ClientError("浏览器验证未完成。可用 /browser 重试；Cloudflare 可能不接受自动化浏览器。")

    async def _browser_request(self, method, path, data):
        try:
            r = await self.page.evaluate("""async ({method,path,data,timeout}) => {
                const controller = new AbortController();
                const timer = setTimeout(() => controller.abort(), timeout);
                try {
                    const opts = {method, credentials:'same-origin', cache:'no-store',
                                  redirect:'error', signal:controller.signal};
                    if (method === 'POST') opts.body = new URLSearchParams(data);
                    const r = await fetch(path, opts);
                    return {status:r.status, text:await r.text(),
                            headers:Object.fromEntries(r.headers), url:r.url};
                } finally {clearTimeout(timer);}
            }""", {"method": method, "path": path, "data": data, "timeout": self.timeout * 1000})
            return Reply(**r)
        except Exception:
            raise NetworkError("浏览器请求失败；请保持验证窗口打开，必要时输入 /browser。") from None

    async def request(self, method, path, data=None):
        if not path.startswith(LOBBY) or urlparse(path).netloc:
            raise ClientError("拒绝访问非 QOJ 斗地主路径。")
        async with self.lock:
            if self.mode == "browser" and not self.page:
                if self.browser_attempted:
                    raise ClientError("浏览器尚未就绪；请先输入 /browser。")
                await self.enable_browser()
            fn = self._browser_request if self.page else self._direct
            reply = await fn(method, path, data)
            if is_challenge(reply):
                if self.mode == "http":
                    raise ClientError("收到 Cloudflare 验证，请输入 /browser 或用 --transport browser 重启。")
                # One automatic handoff; subsequent challenges require /browser, avoiding loops.
                if self.browser_attempted:
                    raise ClientError("Cloudflare 仍要求验证；请输入 /browser 在窗口中完成。")
                await self.enable_browser()
                if method == "GET" or (data or {}).get("action") in ("state", "lobby"):
                    reply = await self._browser_request(method, path, data)
                else:
                    raise NetworkError("已切换浏览器；上次操作未自动重发，请先查看最新状态。")
            return reply

    async def close(self):
        for obj, action in ((self.context, "close"), (self.browser, "close"), (self.pw, "stop")):
            if obj:
                with contextlib.suppress(Exception):
                    await getattr(obj, action)()
        with contextlib.suppress(Exception):
            await asyncio.get_running_loop().run_in_executor(self.executor, self.session.close)
        self.executor.shutdown(wait=False)


class QojClient:
    def __init__(self, transport):
        self.transport = transport
        self.lobby = LobbyInfo()
        self.lobby_status = {}
        self.state = None
        self.token = ""
        self.chat = {}
        self.chat_after = 0
        self.commitments = {}
        self.queue_may_be_active = False

    async def get_html(self, path):
        reply = await self.transport.request("GET", path)
        if not 200 <= reply.status < 300 or is_challenge(reply):
            raise ApiError(f"页面请求失败（HTTP {reply.status}）；可用 /browser 完成验证。", status=reply.status)
        return reply.text

    async def load_lobby(self):
        info = parse_lobby(await self.get_html(LOBBY + "?locale=zh-cn"))
        self.lobby, self.token = info, info.token
        self.lobby_status = {"game": info.game, "queued": None}
        self.state = None

    async def load_game(self, game_id):
        token, data = parse_game(await self.get_html(f"{LOBBY}/game/{int(game_id)}?locale=zh-cn"))
        self.token = token
        self.state = None
        self.chat, self.chat_after, self.commitments = {}, 0, {}
        self.accept(data)

    def accept(self, data):
        if isinstance(data.get("lobby"), dict):
            self.lobby_status = data["lobby"]
            self.queue_may_be_active = bool(self.lobby_status.get("queued"))
        next_state = data.get("state")
        if isinstance(next_state, dict) and not next_state.get("unchanged"):
            if (self.state is None or next_state.get("id") != self.state.get("id")
                    or next_state.get("version", 0) >= self.state.get("version", 0)):
                self.state = next_state
                for i, digest in enumerate(next_state.get("fairness", {}).get("commitments", [])):
                    self.commitments.setdefault(i, digest)
        # unchanged:true must NOT replace the hand/log/phase, but chat still advances.
        for message in data.get("chat") or []:
            if isinstance(message.get("id"), int):
                self.chat[message["id"]] = message
                self.chat_after = max(self.chat_after, message["id"])

    async def call(self, action, **extra):
        body = {"_token": self.token, "action": action}
        if action in ("queue", "cancel", "lobby"):
            body["mode"] = "single"
        else:
            if not self.state:
                raise ClientError("请先进入对局。")
            body.update(game=str(self.state["id"]), version=str(self.state["version"]), chat_after=str(self.chat_after))
        body.update({k: str(v) for k, v in extra.items()})
        if action == "queue":
            self.queue_may_be_active = True
        reply = await self.transport.request("POST", API, body)
        try:
            data = json.loads(reply.text)
            if not isinstance(data, dict):
                raise ValueError
        except (ValueError, TypeError):
            if "/login" in reply.url or 'type="password"' in reply.text:
                raise LoginError("登录已过期，请重新运行输入 Cookie。") from None
            raise ApiError(f"接口未返回 JSON（HTTP {reply.status}）；可输入 /browser 或 /refresh。", status=reply.status) from None
        self.accept(data)  # Error responses may contain a fresh authoritative state.
        if not 200 <= reply.status < 300 or data.get("error"):
            retry_after = reply.headers.get("retry-after", "0")
            raise ApiError(clean(data.get("error") or f"HTTP {reply.status}"), data, reply.status,
                           min(60, int(retry_after)) if str(retry_after).isdigit() else 0)
        return data

    async def refresh_token(self):
        if self.state:
            token, _ = parse_game(await self.get_html(f"{LOBBY}/game/{self.state['id']}?locale=zh-cn"))
            self.token = token
        else:
            self.lobby = parse_lobby(await self.get_html(LOBBY + "?locale=zh-cn"))
            self.token = self.lobby.token


def name_of(state, seat):
    return clean(state["players"][seat]["username"]) if isinstance(seat, int) and 0 <= seat < 3 else "—"


def seat_order(state):
    me = state.get("seat")
    return [(me + 1) % 3, (me + 2) % 3, me] if me is not None else [0, 1, 2]


def to_beat(state):
    return (state.get("last") or {}).get("pattern") if state.get("phase") == "playing" and not state.get("leading") else None


def compact_log(state):
    out = []
    for e in state.get("log", []):
        kind = e.get("kind")
        if kind == "redeal":
            out.append("重新发牌")
            continue
        name = name_of(state, e.get("seat"))
        if kind == "play":
            text = cards_text(e["cards"])
        elif kind == "pass":
            text = "jump"
        elif kind == "bid":
            text = f"叫{e['value']}" if e["value"] else "不叫"
        elif kind == "landlord":
            text = f"地主{e['value']} 底{cards_text(e.get('cards'))}"
        elif kind == "auto_on":
            text = "超时托管" if e.get("timeout") else "托管"
        elif kind == "auto_off":
            text = "取消托管"
        elif kind == "finish":
            text = "出完"
        else:
            text = clean(kind or "事件")
        out.append(f"{name}:{text}" + ("*" if e.get("auto") else ""))
    return out


def pack_logs(entries, width):
    from prompt_toolkit.utils import get_cwidth
    rows, row = [], []
    for entry in entries:
        if row and (len(row) == 3 or get_cwidth("  |  ".join(row + [entry])) > max(20, width - 2)):
            rows.append("  |  ".join(row))
            row = []
        row.append(entry)
    if row:
        rows.append("  |  ".join(row))
    return rows


def counter_text(state):
    left = [4] * 13 + [1, 1]
    for e in state.get("log", []):
        if e.get("kind") == "play":
            for c in e["cards"]:
                left[rank(c)] -= 1
    for c in state.get("hand") or []:
        left[rank(c)] -= 1
    return " ".join(f"{RANKS[r]}:{max(0, left[r])}" for r in range(14, -1, -1))


def fairness_lines(client, detail=False):
    fairness = client.state.get("fairness") or {}
    commits, deals = fairness.get("commitments", []), fairness.get("deals", [])
    if not commits:
        return ["发牌公平性：服务器未提供承诺"]
    if not deals and not detail:
        return [f"发牌公平性：已记录 {len(commits)} 份 SHA-256 承诺，终局校验（/fair 查看）"]
    rows = []
    for i, commitment in enumerate(commits):
        note = "等待终局公开"
        deal = deals[i] if i < len(deals) else None
        if deal:
            digest = hashlib.sha256((",".join(map(str, deal["deck"])) + "|" + deal["salt"]).encode()).hexdigest()
            valid_deck = sorted(deal["deck"]) == list(range(54))
            ok = digest == commitment == client.commitments.get(i, commitment) == deal.get("commitment")
            note = "校验通过" if ok and valid_deck else "校验不一致！"
        rows.append(f"发牌 {i + 1}：{commitment if detail else commitment[:12] + '…'} {note}")
        if detail and deal:
            rows += ["deck=" + ",".join(map(str, deal["deck"])), "salt=" + deal["salt"]]
    return rows


HELP = """输入牌面即可出牌：789XJQKA；- 过；其余普通文本发送聊天。
X / T / 10 = 10，S / 小王 = 小王，D / 大王 = 大王；大小写均可。
叫分阶段：0 或 - 不叫；1 / 2 / 3 叫分。出牌阶段的 2、3 是牌。
牌面字符串总是出牌尝试；牌型不合法、缺牌或管不上时显示错误。
/say 文本 强制聊天（例如 /say 333）；/play 牌面 强制出牌。
同一手有多种牌型时：按 牌面#编号 出牌，或 /choose 编号。
/hint 提示（只列出建议，不会出牌）；/auto on|off 托管开关。
/mute 屏蔽/恢复他人聊天；/sort 切换手牌顺序；/fair 发牌校验详情。
/initial 终局开局手牌；/log 全部日志；/chat 全部聊天；/help 本说明。
/close 关闭详情；详情用 PgUp/PgDn 滚动；F2/F3 滚动日志/聊天，Esc 回到底部。
/cancel 取消匹配；/home 回大厅；/again 终局后再匹配一局；/refresh 同步。
/browser 打开/重新验证浏览器；/quit 或 Ctrl-C 退出。
大厅：1 单局匹配，2 记分比赛（本版未实现，不会加入队列）。
日志中的 * 表示服务器托管操作；不展示计时器。"""


class TerminalUI:
    def __init__(self, client, demo=False, input=None, output=None, show_chat=True):
        from prompt_toolkit.application import Application
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.document import Document
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import HSplit, VSplit, Layout, Window
        from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
        from prompt_toolkit.layout.containers import ConditionalContainer
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.styles import Style
        from prompt_toolkit.widgets import TextArea
        self.client, self.demo = client, demo
        self.notice = "正在连接…" if not demo else "离线演示；不会发送任何网络请求。"
        self.connected = demo
        self.busy = False
        self.quitting = False
        self.muted = False
        self.desc = False
        self.detail = ""
        self.pending = None
        self.scroll = {"log": 0, "chat": 0}
        self.Document = Document
        self.op_lock = asyncio.Lock()
        self.commands = asyncio.Queue(maxsize=1)
        self.wakeup = asyncio.Event()
        self.details = TextArea(read_only=True, scrollbar=True, wrap_lines=True)
        kb = KeyBindings()

        @kb.add("c-c")
        @kb.add("c-d")
        def quit_key(event):
            self.quitting = True
            self.notice = "正在退出…"
            self.app.exit()

        @kb.add("escape")
        def escape(event):
            self.detail = ""
            self.scroll = {"log": 0, "chat": 0}
            event.app.layout.focus(self.input_control)

        @kb.add("f2")
        def scroll_log(event):
            self.scroll["log"] += 3
            event.app.invalidate()

        @kb.add("f3", filter=Condition(lambda: show_chat))
        def scroll_chat(event):
            self.scroll["chat"] += 3
            event.app.invalidate()

        @kb.add("pageup")
        def page_up(event):
            self.details.buffer.cursor_up(count=10)

        @kb.add("pagedown")
        def page_down(event):
            self.details.buffer.cursor_down(count=10)

        def submit(buffer):
            text = buffer.text.strip()
            if not text:
                return False
            if self.busy or not self.commands.empty():
                self.set_notice("上一条输入仍在处理；本行已保留。")
                return True
            self.commands.put_nowait(text)
            return False

        self.buffer = Buffer(accept_handler=submit, multiline=False)
        self.input_control = BufferControl(self.buffer)
        game = Condition(lambda: self.client.state is not None and not self.detail)
        home = Condition(lambda: self.client.state is None and not self.detail)
        details = Condition(lambda: bool(self.detail))
        self.log_window = Window(FormattedTextControl(self.render_log), height=lambda: self.log_height(), wrap_lines=True)
        self.chat_window = Window(FormattedTextControl(self.render_chat), height=lambda: self.chat_height(), wrap_lines=True)
        self.board_window = Window(FormattedTextControl(self.render_board), wrap_lines=True)
        body = HSplit([
            Window(FormattedTextControl(self.render_header), height=1, style="class:header"),
            ConditionalContainer(Window(FormattedTextControl(self.render_home), wrap_lines=True), home),
            ConditionalContainer(HSplit([
                Window(FormattedTextControl(" 出牌日志（每行最多三条；F2 向前 / Esc 最新）"), height=1, style="class:section"),
                self.log_window,
                *([Window(FormattedTextControl(" 聊天（F3 向前 / Esc 最新）"), height=1, style="class:section"),
                   self.chat_window] if show_chat else []),
                Window(FormattedTextControl(" 牌局 · 从上到下按出牌顺序循环"), height=1, style="class:section"),
                self.board_window,
            ]), game),
            ConditionalContainer(self.details, details),
            ConditionalContainer(Window(FormattedTextControl(self.render_hand), height=lambda: 2 if self.small_screen() else 3, wrap_lines=True, style="class:hand"), Condition(lambda: self.client.state is not None)),
            Window(FormattedTextControl(lambda: clean(self.notice)), height=lambda: 1 if self.small_screen() else 2, wrap_lines=True, style="class:notice"),
            Window(FormattedTextControl(self.render_prompt), height=1, wrap_lines=True),
            VSplit([Window(FormattedTextControl("> "), width=2, height=1),
                    Window(self.input_control, height=1, wrap_lines=False, always_hide_cursor=False)], style="class:input"),
        ])
        self.app = Application(layout=Layout(body, self.input_control), key_bindings=kb,
                               full_screen=True, mouse_support=False, input=input, output=output,
                               style=Style.from_dict({"header": "bg:#17324d #ffffff bold", "section": "#66c2ff bold",
                                                      "hand": "#ffdc75 bold", "notice": "#a7dba5", "input": "bg:#203040 #ffffff"}))

    def set_notice(self, text):
        self.notice = clean(text)
        self.app.invalidate()

    def small_screen(self):
        return self.app.output.get_size().rows < 30

    def log_height(self):
        return 2 if self.small_screen() else 4

    def chat_height(self):
        return 1 if self.small_screen() else 3

    def render_header(self):
        state = self.client.state
        place = f"对局 #{state['id']}" if state else "大厅"
        link = "离线演示" if self.demo else self.client.transport.label
        return f" QOJ 斗地主 | 单局匹配 | {place} | {link} | /help"

    def render_home(self):
        info, status = self.client.lobby, self.client.lobby_status
        rows = ["", f"  玩家：{clean(info.username) or '—'}    我的 Rating：{info.rating}", ""]
        rows += ["  " + "    ".join(f"{k}：{info.stats.get(k, '—')}" for k in ("总积分", "对局", "胜率", "地主胜率")), ""]
        if status.get("queued"):
            mode = status["queued"]
            size = (status.get("queue_size") or {}).get(mode, "—")
            rows += [f"  {'单局' if mode == 'single' else '记分比赛'}匹配中 · 队列 {size} 人",
                     f"  同桌积分差 ≤ {status.get('window', '—')} · 我的积分 {status.get('score', '—')}",
                     "  /cancel 取消匹配"]
        elif status.get("game"):
            rows += [f"  已有进行中的对局 #{status['game']} · 输入 1 返回对局"]
        else:
            rows += ["  1  单局匹配", "  2  记分比赛（后续实现）"]
        rows += ["", "  /refresh 刷新资料    /browser 浏览器验证    /quit 退出"]
        return "\n".join(rows)

    def _tail(self, rows, key, height):
        end = max(1, len(rows) - self.scroll[key])
        return "\n".join(rows[max(0, end - height):end]) if rows else "（暂无）"

    def render_log(self):
        state = self.client.state
        if not state:
            return ""
        width = self.app.output.get_size().columns
        return self._tail(pack_logs(compact_log(state), width), "log", self.log_height())

    def render_chat(self):
        state = self.client.state
        if not state:
            return ""
        rows = [f"{clean(m['username'])}: {clean(m['text'])}" for _, m in sorted(self.client.chat.items())
                if not self.muted or m.get("seat") == state.get("seat")]
        return self._tail(rows, "chat", self.chat_height())

    def player_row(self, seat, mine=False):
        s = self.client.state
        p = s["players"][seat]
        marker = "> " if s.get("turn") == seat and s.get("phase") != "finished" else "  "
        tags = (" (地主)" if s.get("landlord") == seat else "") + (" [我]" if mine else "")
        tags += " [托管]" if p.get("auto") else ""
        bid = p.get("bid")
        bid_text = (" · 不叫" if bid == 0 else f" · 叫{bid}分") if bid is not None and s.get("phase") == "bidding" else ""
        table = (s.get("table") or [None] * 3)[seat]
        played = "jump" if table == "pass" else (cards_text(table.get("cards")) + " " + describe(table.get("pattern")) if table else "—")
        row = f"{marker}{name_of(s, seat)}{tags} · 剩 {p['count']} 张{bid_text} · 桌面 {played}"
        if not mine:
            hands = s.get("hands")
            row += "\n    手牌：" + (cards_text(hands[seat], True, self.desc) if hands is not None else "未公开")
        return row

    def render_board(self):
        s = self.client.state
        if not s:
            return ""
        phase = {"bidding": "叫分中", "playing": "出牌中", "finished": "已结束"}.get(s["phase"], s["phase"])
        who = name_of(s, s.get("turn"))
        rows = [f"{phase} · 轮到 {who} · 底分 {s.get('bid', 0)} · ×{s.get('multiplier', 1)} · 炸弹 {s.get('bombs', 0)} · 流局 {s.get('redeals', 0)}",
                f"底牌：{cards_text(s['bottom']) if s.get('bottom') is not None else '未公开'}" + (" · 自由出牌" if s.get("leading") else ""),
                "未出现（不含我的手牌）：" + counter_text(s)]
        for seat in seat_order(s)[:-1]:
            rows.append(self.player_row(seat))
        r = s.get("result")
        if r:
            tags = (" · 春天" if r.get("spring") else "") + (" · 反春" if r.get("anti_spring") else "")
            score = "  ".join(f"{name_of(s, i)} {d:+}" for i, d in enumerate(r["deltas"]))
            rows += [f"结算：{'地主胜' if r['landlord_won'] else '农民胜'}{tags} · {r['base']}×{r['multiplier']} · {score}"]
        rows += fairness_lines(self.client)
        return "\n".join(rows)

    def render_hand(self):
        s = self.client.state
        if not s:
            return ""
        return self.player_row(s["seat"], True) + "\n我的手牌：" + cards_text(s.get("hand"), True, self.desc)

    def render_prompt(self):
        s = self.client.state
        if not s:
            return "输入 1 开始单局；2 记分比赛（未实现）。"
        if s["phase"] == "finished":
            return "本局结束：/again 再来一局；/home 大厅；仍可聊天。"
        if s["phase"] == "bidding":
            return ("必须叫分：" if s.get("must_bid") else "叫分：0 或 - 不叫；") + "1 / 2 / 3 叫分；其余文本聊天。"
        return "输入牌面出牌（X=10 S=小王 D=大王）；- 过；其余文本聊天。 /hint 提示"

    def show_detail(self, title, lines):
        self.detail = title
        text = title + "（PgUp/PgDn 滚动，Esc 关闭）\n\n" + "\n".join(lines)
        self.details.buffer.set_document(self.Document(text, 0), bypass_readonly=True)

    def require_turn(self, phase):
        s = self.client.state
        if not s or s["phase"] != phase:
            raise ClientError("当前不在" + ("叫分" if phase == "bidding" else "出牌") + "阶段。")
        if s.get("turn") != s.get("seat"):
            raise ClientError("还没有轮到你。")
        return s

    async def send_chat(self, text):
        if not text.strip():
            raise ClientError("聊天不能为空。")
        if len(text.encode("utf-16-le")) // 2 > 60:
            raise ClientError("聊天最多 60 个 UTF-16 字符，与网页输入框一致。")
        await self.client.call("chat", text=text.strip())
        self.set_notice("聊天已发送。")

    async def play(self, text, choice_index=None):
        s = self.require_turn("playing")
        ranks = parse_ranks(text)
        if ranks is None:
            raise ClientError("牌面格式错误；示例 789XJQKA。")
        cards = select_cards(s.get("hand"), ranks)
        readings = classify(cards)
        options = [p for p in readings if beats(p, to_beat(s))]
        if not options:
            raise ClientError("这不是合法牌型。" if not readings else "这手牌管不上。")
        if len(options) > 1 and choice_index is None:
            self.pending = (s["id"], s["version"], text, options)
            choices = "  ".join(f"{i + 1}:{describe(p)}" for i, p in enumerate(options))
            self.set_notice(f"请选择牌型 {choices}；输入 /choose 编号，或 {text}#编号")
            return
        index = 0 if choice_index is None else choice_index - 1
        if not 0 <= index < len(options):
            raise ClientError("牌型编号无效。")
        await self.client.call("play", cards=",".join(map(str, cards)), choice=pattern_key(options[index]))
        self.pending = None
        self.set_notice("已出 " + cards_text(cards))

    async def enter_or_queue(self):
        status = self.client.lobby_status
        if status.get("queued"):
            raise ClientError("已经在匹配中。/cancel 可以取消。")
        if status.get("game"):
            await self.client.load_game(status["game"])
        else:
            await self.client.call("queue")
            if self.client.lobby_status.get("game"):
                await self.client.load_game(self.client.lobby_status["game"])
        self.set_notice("已进入对局。" if self.client.state else "已加入单局匹配。")

    async def command(self, line):
        cmd, _, arg = line.partition(" ")
        cmd = cmd.lower()
        s = self.client.state
        if cmd in ("/quit", "/exit"):
            self.quitting = True
            self.app.exit()
            return
        if cmd in ("/help", "/?"):
            self.show_detail("操作说明", HELP.splitlines())
            return
        if cmd == "/close":
            self.detail = ""
            return
        if cmd == "/sort":
            self.desc = not self.desc
            return
        if cmd == "/mute":
            self.muted = not self.muted
            self.set_notice("已屏蔽他人聊天。" if self.muted else "已恢复他人聊天。")
            return
        if cmd in ("/fair", "/initial", "/log", "/chat", "/hint"):
            if not s:
                raise ClientError("请先进入对局。")
            if cmd == "/fair":
                self.show_detail("发牌公平性", fairness_lines(self.client, True))
            elif cmd == "/log":
                self.show_detail("全部日志", pack_logs(compact_log(s), self.app.output.get_size().columns))
            elif cmd == "/chat":
                self.show_detail("全部聊天", [f"{clean(m['username'])}: {clean(m['text'])}" for _, m in sorted(self.client.chat.items())
                                             if not self.muted or m.get("seat") == s.get("seat")])
            elif cmd == "/initial":
                deals = (s.get("fairness") or {}).get("deals")
                if not deals:
                    raise ClientError("开局牌序在终局后由服务器公开。")
                deck = deals[-1]["deck"]
                self.show_detail("最后一次发牌的开局手牌", [name_of(s, i) + ": " + cards_text(deck[17*i:17*i+17], True) for i in seat_order(s)] + ["底牌：" + cards_text(deck[51:])])
            else:
                self.require_turn("playing")
                rows = [cards_text(c) for c in hints(s["hand"], to_beat(s))]
                self.set_notice("提示：" + (" / ".join(rows[:12]) if rows else "没有能管上的牌。"))
            return
        if self.demo:
            self.set_notice("离线演示只供查看；真实操作请去掉 --demo。")
            return
        if cmd == "/browser":
            async with self.client.transport.lock:
                await self.client.transport.enable_browser()
            if self.client.token:
                await self.client.refresh_token()
            else:
                await self.client.load_lobby()
            await self.client.call("state" if s else "lobby")
            self.connected = True
            return
        if cmd == "/refresh":
            if self.client.token:
                await self.client.refresh_token()
            else:
                await self.client.load_lobby()
            await self.client.call("state" if self.client.state else "lobby")
            self.connected = True
            self.set_notice("状态已刷新。")
            return
        if not self.connected:
            raise ClientError("连接尚未就绪；可用 /refresh 或 /browser 重试。")
        if cmd in ("/home", "/again"):
            if s and s["phase"] != "finished":
                raise ClientError("本局仍在进行，请完成后返回大厅。")
            if self.client.lobby_status.get("queued") and not s:
                await self.client.call("cancel")
            await self.client.load_lobby()
            await self.client.call("lobby")
            self.detail, self.pending = "", None
            if cmd == "/again":
                await self.enter_or_queue()
            return
        if cmd == "/cancel":
            if s:
                raise ClientError("已经开始对局，不能取消匹配。")
            await self.client.call("cancel")
            self.set_notice("匹配已取消。" if not self.client.lobby_status.get("game") else "取消前已匹配成功；输入 1 返回对局。")
            return
        if not s:
            if line == "1":
                await self.enter_or_queue()
            elif line == "2":
                self.set_notice("本版只实现单局匹配；记分比赛暂不可用。")
            else:
                raise ClientError("大厅请输入 1 或 2，或 /help。")
            return
        if cmd == "/say":
            await self.send_chat(arg)
            return
        if cmd == "/auto":
            if s["phase"] == "finished":
                raise ClientError("对局已经结束。")
            on = arg.lower()
            if on not in ("on", "off", "1", "0"):
                raise ClientError("用法：/auto on 或 /auto off")
            await self.client.call("auto", on="1" if on in ("on", "1") else "0")
            self.set_notice("托管状态已更新。")
            return
        if cmd == "/choose":
            if not arg.isdigit() or not self.pending:
                raise ClientError("当前没有待选择的牌型，或编号无效。")
            game, version, text, _ = self.pending
            if game != s["id"] or version != s["version"]:
                self.pending = None
                raise ClientError("牌局已变化，请重新输入要出的牌。")
            await self.play(text, int(arg))
            return
        if cmd == "/play":
            line = arg
        elif line.startswith("/"):
            raise ClientError("未知命令；输入 /help 查看。用 /say 可发送以 / 开头的聊天。")
        if s["phase"] == "bidding" and line in ("-", "0", "1", "2", "3"):
            self.require_turn("bidding")
            value = 0 if line == "-" else int(line)
            if (value == 0 and s.get("must_bid")) or (value and value <= s.get("bid", 0)):
                raise ClientError("必须叫更高的分数。" if value else "本轮必须叫分，不能不叫。")
            await self.client.call("bid", value=value)
            self.set_notice("已不叫。" if not value else f"已叫 {value} 分。")
        elif line == "-":
            self.require_turn("playing")
            if s.get("leading"):
                raise ClientError("轮到你自由出牌，不能过。")
            await self.client.call("pass")
            self.set_notice("已过。")
        else:
            match = re.fullmatch(r"(.+)#([1-9]\d*)", line)
            text, choice = (match[1], int(match[2])) if match else (line, None)
            if parse_ranks(text) is not None or cmd == "/play":
                await self.play(text, choice)
            else:
                await self.send_chat(line)

    async def command_loop(self):
        while True:
            line = await self.commands.get()
            self.busy = True
            try:
                async with self.op_lock:
                    await self.command(line)
            except NetworkError as e:
                self.set_notice(f"{e} 上次操作结果不确定，未重发；请等待轮询并核对牌局/聊天。")
            except ClientError as e:
                self.set_notice(str(e))
            except Exception as e:
                self.set_notice(f"处理失败（{type(e).__name__}）；可 /refresh，同步失败请反馈页面结构。")
            finally:
                self.busy = False
                self.wakeup.set()
                self.app.invalidate()

    async def poll_loop(self):
        failures = 0
        while True:
            delay = 0.9 if self.client.state and self.client.state.get("phase") != "finished" else 3.0
            if not self.client.state and self.client.lobby_status.get("queued"):
                delay = 1.5
            elif not self.client.state:
                delay = 5.0
            self.wakeup.clear()
            try:
                await asyncio.wait_for(self.wakeup.wait(), timeout=delay if not failures else min(10, 1.5 * 2**min(failures, 4)))
            except asyncio.TimeoutError:
                pass
            if not self.connected or self.busy:
                continue
            try:
                async with self.op_lock:
                    was_queued = self.client.queue_may_be_active or bool(self.client.lobby_status.get("queued"))
                    action = "state" if self.client.state else "lobby"
                    await self.client.call(action)
                    if not self.client.state and was_queued and self.client.lobby_status.get("game"):
                        await self.client.load_game(self.client.lobby_status["game"])
                        self.set_notice("匹配成功，已进入对局。")
                    if failures:
                        self.set_notice("连接已恢复；已同步最新状态。")
                    failures = 0
                    self.app.invalidate()
            except LoginError as e:
                self.connected = False
                self.set_notice(str(e))
            except ClientError as e:
                failures += 1
                self.set_notice(f"状态同步失败：{e}（轮询会重试）")
                if isinstance(e, ApiError) and e.retry_after:
                    await asyncio.sleep(e.retry_after)
            except Exception as e:
                failures += 1
                self.set_notice(f"状态格式发生变化（{type(e).__name__}）；请提供最新页面 HTML。")

    async def connect(self):
        try:
            async with self.op_lock:
                await self.client.load_lobby()
                await self.client.call("lobby")
                if self.client.lobby_status.get("game"):
                    await self.client.load_game(self.client.lobby_status["game"])
                self.connected = True
                self.set_notice("连接成功，请输入操作。")
        except ClientError as e:
            self.set_notice(str(e) + " 可用 /refresh 或 /browser 重试。")
        except Exception as e:
            self.set_notice(f"初始化失败（{type(e).__name__}）；请提供最新 HTML 以核对结构。")

    async def run(self):
        tasks = []

        def start():
            tasks.append(asyncio.create_task(self.command_loop()))
            if not self.demo:
                tasks.extend([asyncio.create_task(self.connect()), asyncio.create_task(self.poll_loop())])
        try:
            await self.app.run_async(pre_run=start)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def demo_client():
    c = QojClient(None)
    c.lobby = LobbyInfo(username="me", rating="1500", stats={"总积分": "1", "对局": "20", "胜率": "45.0%", "地主胜率": "75.0%"})
    hand = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 53]
    c.accept({"state": {"id": 25012, "version": 8, "phase": "playing", "seat": 2,
        "players": [{"username": "jia", "count": 17, "bid": 0, "auto": False},
                    {"username": "yi", "count": 17, "bid": 0, "auto": False},
                    {"username": "me", "count": len(hand), "bid": 2, "auto": False}],
        "turn": 2, "landlord": 2, "bottom": [48, 52, 53], "hand": hand, "hands": None,
        "bid": 2, "multiplier": 1, "bombs": 0, "redeals": 0, "leading": True, "last": None,
        "table": ["pass", "pass", {"cards": [1, 2, 3, 5, 6], "pattern": {"type": "trio2", "rank": 0, "len": 1}}],
        "log": [{"kind": "bid", "seat": 2, "value": 2}, {"kind": "landlord", "seat": 2, "value": 2, "cards": [48, 52, 53]},
                {"kind": "play", "seat": 2, "cards": [1, 2, 3, 5, 6]}, {"kind": "pass", "seat": 0}, {"kind": "pass", "seat": 1}],
        "result": None, "fairness": {"commitments": ["0" * 64]}},
        "chat": [{"id": 1, "seat": 0, "username": "jia", "text": "大家好"}, {"id": 2, "seat": 2, "username": "me", "text": "开始吧"}]})
    return c


async def main_async(args):
    if args.demo:
        await TerminalUI(demo_client(), demo=True).run()
        return
    cookie = parse_cookie(getpass.getpass("请输入 __Host-UOJSESSID（输入不回显）："))
    transport = Transport(cookie, args.transport, args.browser, args.timeout)
    client = QojClient(transport)
    ui = TerminalUI(client)
    transport.notify = ui.set_notice
    try:
        await ui.run()
    finally:
        # No move, chat or automatic play is sent on quit. Cancel only our lobby queue.
        if (client.queue_may_be_active or client.lobby_status.get("queued")) and not client.state:
            try:
                await asyncio.wait_for(client.call("cancel"), timeout=args.timeout + 1)
                if client.lobby_status.get("game"):
                    print("取消前已匹配成功；重新运行 CLI 或在网页返回进行中的对局。")
            except Exception:
                print("未能确认取消匹配，请在 QOJ 网页检查队列状态。")
        await transport.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo", action="store_true", help="离线界面演示，不需要 Cookie")
    parser.add_argument("--transport", choices=("auto", "http", "browser"), default="auto", help="auto: 浏览器指纹 HTTP，遇验证回退浏览器")
    parser.add_argument("--browser", choices=("auto", "chromium", "chrome", "msedge"), default="auto")
    parser.add_argument("--timeout", type=int, default=12, help="单次网络请求超时秒数（3–60）")
    parser.add_argument('--package', nargs='?', type=Path, const=ROOT/'qoj_bot_client.zip', help='打包当前独立客户端；可指定 ZIP 路径')
    args = parser.parse_args()
    if args.package:
        package_client(args.package)
        return 0
    if not 3 <= args.timeout <= 60:
        parser.error("--timeout 必须为 3–60")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        asyncio.run(main_async(args))
    except (KeyboardInterrupt, EOFError):
        pass
    except ModuleNotFoundError as e:
        print(f"缺少依赖 {e.name}。请运行：python -m pip install -r requirements.txt", file=sys.stderr)
        return 1
    except ClientError as e:
        print(clean(e), file=sys.stderr)
        return 1
    return 0


# Complete legal actions and validation

@dataclass(frozen=True)
class Action:
    kind: str
    ranks: tuple = ()
    pattern: tuple = ()
    value: int = 0

    @property
    def reading(self):
        return dict(zip(('type', 'rank', 'len'), self.pattern)) if self.pattern else None

    def to_dict(self):
        return {'kind': self.kind, 'ranks': list(self.ranks),
                'pattern': list(self.pattern), 'value': self.value}

    @classmethod
    def from_dict(cls, value):
        return cls(value['kind'], tuple(value.get('ranks', ())),
                   tuple(value.get('pattern', ())), int(value.get('value', 0)))


def current_log(s):
    log = s.get('log', [])
    for i in range(len(log)-1, -1, -1):
        if log[i].get('kind') == 'redeal':
            return log[i+1:]
    return log


def minimal_action(s):
    """O(hand size) emergency action, without enumerating combinations."""
    if s.get('phase') == 'bidding':
        return Action('bid', value=int(s.get('bid', 0))+1 if s.get('must_bid') else 0)
    if s.get('phase') != 'playing' or not s.get('hand'):
        raise ValueError('当前没有可决策的手牌。')
    if not s.get('leading'):
        return Action('pass')
    r = min(map(rank, s['hand']))
    return Action('play', (r,), ('single', r, 1))


def _wings(counts, excluded, k, unit):
    available = [(r, min(n, 3) if unit == 1 else int(n >= 2))
                 for r, n in enumerate(counts) if n and r not in excluded and (unit == 1 or r < 13)]
    def visit(i, left, acc):
        if not left:
            if not (13 in acc and 14 in acc):
                yield tuple(acc)
            return
        if i == len(available) or sum(cap for _, cap in available[i:]) < left:
            return
        r, cap = available[i]
        for take in range(min(left, cap)+1):
            yield from visit(i+1, left-take, acc+[r]*(take*unit))
    yield from visit(0, k, [])


@lru_cache(maxsize=256)
def _rank_actions(counts):
    candidates = set()
    def add(rs): candidates.add(tuple(sorted(rs)))
    present = [r for r, n in enumerate(counts) if n]
    for r in present:
        for k in range(1, min(4, counts[r])+1): add([r]*k)
        if counts[r] >= 3:
            for wing in _wings(counts, {r}, 1, 1): add([r]*3+list(wing))
            for wing in _wings(counts, {r}, 1, 2): add([r]*3+list(wing))
        if counts[r] == 4:
            for wing in _wings(counts, {r}, 2, 1): add([r]*4+list(wing))
            for wing in _wings(counts, {r}, 2, 2): add([r]*4+list(wing))
    if counts[13] and counts[14]: add([13, 14])
    for unit, minimum in ((1, 5), (2, 3), (3, 2)):
        for start in range(12):
            for end in range(start, 12):
                if counts[end] < unit: break
                k = end-start+1
                if k < minimum: continue
                body = [r for r in range(start, end+1) for _ in range(unit)]
                add(body)
                if unit == 3:
                    for wingunit in (1, 2):
                        if len(body)+k*wingunit > sum(counts): continue
                        for wing in _wings(counts, set(range(start, end+1)), k, wingunit):
                            add(body+list(wing))
    out = []
    for rs in sorted(candidates, key=lambda rs: (len(rs), rs)):
        synthetic = []
        for r, n in Counter(rs).items():
            synthetic.extend(range(r*4, r*4+n) if r < 13 else [r+39])
        for p in classify(synthetic):
            out.append(Action('play', rs, (p['type'], p['rank'], p['len'])))
    return tuple(out)


def legal_actions(s):
    if s.get('phase') == 'bidding':
        allowed = list(range(int(s.get('bid', 0))+1, 4))
        if not s.get('must_bid'): allowed.insert(0, 0)
        return [Action('bid', value=v) for v in allowed]
    if s.get('phase') != 'playing': return []
    c = Counter(map(rank, s['hand']))
    target = to_beat(s)
    out = [a for a in _rank_actions(tuple(c[r] for r in range(15))) if beats(a.reading, target)]
    if not s.get('leading'): out.append(Action('pass'))
    return out


def validate_action(s, a):
    if s.get('seat') != s.get('turn'): raise ValueError('还没有轮到你。')
    if a.kind == 'bid':
        if s.get('phase') != 'bidding' or a not in legal_actions(s): raise ValueError('叫分不合法。')
        return []
    if s.get('phase') != 'playing': raise ValueError('当前不在出牌阶段。')
    if a.kind == 'pass':
        if s.get('leading'): raise ValueError('自由出牌时不能过。')
        return []
    if a.kind != 'play': raise ValueError('未知动作。')
    cards = select_cards(s['hand'], a.ranks)
    if a.reading not in classify(cards) or not beats(a.reading, to_beat(s)):
        raise ValueError('牌型不合法或管不上。')
    return cards


def action_text(a):
    if a.kind == 'bid': return f'叫 {a.value} 分' if a.value else '不叫'
    if a.kind == 'pass': return '过'
    return ''.join(RANKS[r] for r in a.ranks)+' · '+describe(a.reading)


# Offline dealing, bidding and additive scoring

DEFAULT_RULES = {'schema': 1, 'scoring': 'additive', 'force_after_redeals': 3,
                 'redeal_start': 'random', 'ambiguous': 'explicit_choice'}


class Game:
    def __init__(self, seed=None, rules=None, game_id=1):
        self.rng = random.Random(seed)
        self.rules = {**DEFAULT_RULES, **(rules or {})}
        self.id, self.version, self.redeals, self.log = game_id, 0, 0, []
        self.first = self.rng.randrange(3)
        self._deal()

    def _deal(self):
        deck = list(range(54)); self.rng.shuffle(deck)
        self.hands = [sorted(deck[i*17:(i+1)*17]) for i in range(3)]
        self.bottom = deck[51:]
        self.turn, self.phase = self.first, 'bidding'
        self.bids, self.bid_count, self.bid, self.landlord = [-1]*3, 0, 0, None
        self.multiplier, self.bombs = 1, 0
        self.play_counts = [0, 0, 0]
        self.leading, self.last, self.passes, self.result = True, None, 0, None

    @property
    def must_bid(self):
        return (self.phase == 'bidding' and self.bid_count == 2 and self.bid == 0
                and self.redeals >= self.rules['force_after_redeals'])

    def observation(self, seat=None):
        seat = self.turn if seat is None else seat
        return {'id': self.id, 'version': self.version, 'phase': self.phase,
                'seat': seat, 'turn': self.turn, 'hand': self.hands[seat].copy(),
                'players': [{'username': f'玩家{i+1}', 'count': len(self.hands[i]),
                             'bid': self.bids[i], 'landlord': i == self.landlord, 'auto': False}
                            for i in range(3)],
                'landlord': self.landlord, 'bottom': self.bottom.copy() if self.landlord is not None else [],
                'bid': self.bid, 'multiplier': self.multiplier, 'bombs': self.bombs,
                'redeals': self.redeals, 'must_bid': self.must_bid,
                'leading': self.leading, 'last': self.last, 'log': self.log.copy(), 'result': self.result}

    def apply(self, a):
        cards = validate_action(self.observation(), a)
        seat = self.turn; self.version += 1
        if a.kind == 'bid':
            self.log.append({'kind': 'bid', 'seat': seat, 'value': a.value})
            self.bids[seat] = a.value; self.bid_count += 1
            if a.value: self.bid, self.landlord = a.value, seat
            if a.value == 3 or (self.bid_count == 3 and self.bid):
                self.phase, self.turn = 'playing', self.landlord
                self.hands[self.landlord] = sorted(self.hands[self.landlord]+self.bottom)
                self.log.append({'kind': 'landlord', 'seat': self.landlord, 'value': self.bid, 'cards': self.bottom.copy()})
            elif self.bid_count == 3:
                self.redeals += 1
                self.log.append({'kind': 'redeal'})
                if self.rules['redeal_start'] == 'random': self.first = self.rng.randrange(3)
                self._deal()
            else: self.turn = (seat+1)%3
            return
        if a.kind == 'pass':
            self.log.append({'kind': 'pass', 'seat': seat}); self.passes += 1
            if self.passes == 2:
                self.leading, self.turn, self.passes = True, self.last['seat'], 0
            else: self.turn = (seat+1)%3
            return
        for card in cards: self.hands[seat].remove(card)
        self.play_counts[seat] += 1
        self.last = {'seat': seat, 'cards': cards, 'pattern': a.reading}
        self.log.append({'kind': 'play', **self.last})
        self.leading, self.passes = False, 0
        if a.reading['type'] in ('bomb', 'rocket'):
            self.bombs += 1; self.multiplier += 1
        if not self.hands[seat]:
            won = seat == self.landlord
            spring = won and all(self.play_counts[i] == 0 for i in range(3) if i != self.landlord)
            anti = not won and self.play_counts[self.landlord] == 1
            self.multiplier += int(spring or anti)
            farm = -self.bid*self.multiplier if won else self.bid*self.multiplier
            deltas = [farm]*3; deltas[self.landlord] = -2*farm
            self.result = {'landlord_won': won, 'spring': spring, 'anti_spring': anti,
                           'base': self.bid, 'multiplier': self.multiplier, 'deltas': deltas}
            self.phase = 'finished'
            self.log.append({'kind': 'finish', 'seat': seat})
        else: self.turn = (seat+1)%3


# DouZero observation encoding (Apache-2.0)

# Vendored DouZero feature encoding. See README.md and LICENSE.txt.


Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

deck = []
for i in range(3, 15):
    deck.extend([i for _ in range(4)])
deck.extend([17 for _ in range(4)])
deck.extend([20, 30])

def get_obs(infoset):
    """
    This function obtains observations with imperfect information
    from the infoset. It has three branches since we encode
    different features for different positions.
    
    This function will return dictionary named `obs`. It contains
    several fields. These fields will be used to train the model.
    One can play with those features to improve the performance.

    `position` is a string that can be landlord/landlord_down/landlord_up

    `x_batch` is a batch of features (excluding the hisorical moves).
    It also encodes the action feature

    `z_batch` is a batch of features with hisorical moves only.

    `legal_actions` is the legal moves

    `x_no_action`: the features (exluding the hitorical moves and
    the action features). It does not have the batch dim.

    `z`: same as z_batch but not a batch.
    """
    if infoset.player_position == 'landlord':
        return _get_obs_landlord(infoset)
    elif infoset.player_position == 'landlord_up':
        return _get_obs_landlord_up(infoset)
    elif infoset.player_position == 'landlord_down':
        return _get_obs_landlord_down(infoset)
    else:
        raise ValueError('')

def _get_one_hot_array(num_left_cards, max_num_cards):
    """
    A utility function to obtain one-hot endoding
    """
    one_hot = np.zeros(max_num_cards)
    one_hot[num_left_cards - 1] = 1

    return one_hot

def _cards2array(list_cards):
    """
    A utility function that transforms the actions, i.e.,
    A list of integers into card matrix. Here we remove
    the six entries that are always zero and flatten the
    the representations.
    """
    if len(list_cards) == 0:
        return np.zeros(54, dtype=np.int8)

    matrix = np.zeros([4, 13], dtype=np.int8)
    jokers = np.zeros(2, dtype=np.int8)
    counter = Counter(list_cards)
    for card, num_times in counter.items():
        if card < 20:
            matrix[:, Card2Column[card]] = NumOnes2Array[num_times]
        elif card == 20:
            jokers[0] = 1
        elif card == 30:
            jokers[1] = 1
    return np.concatenate((matrix.flatten('F'), jokers))

def _action_seq_list2array(action_seq_list):
    """
    A utility function to encode the historical moves.
    We encode the historical 15 actions. If there is
    no 15 actions, we pad the features with 0. Since
    three moves is a round in DouDizhu, we concatenate
    the representations for each consecutive three moves.
    Finally, we obtain a 5x162 matrix, which will be fed
    into LSTM for encoding.
    """
    action_seq_array = np.zeros((len(action_seq_list), 54))
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _cards2array(list_cards)
    action_seq_array = action_seq_array.reshape(5, 162)
    return action_seq_array

def _process_action_seq(sequence, length=15):
    """
    A utility function encoding historical moves. We
    encode 15 moves. If there is no 15 moves, we pad
    with zeros.
    """
    sequence = sequence[-length:].copy()
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        empty_sequence.extend(sequence)
        sequence = empty_sequence
    return sequence

def _get_one_hot_bomb(bomb_num):
    """
    A utility function to encode the number of bombs
    into one-hot representation.
    """
    one_hot = np.zeros(15)
    one_hot[bomb_num] = 1
    return one_hot

def _get_obs_landlord(infoset):
    """
    Obttain the landlord features. See Table 4 in
    https://arxiv.org/pdf/2106.06135.pdf
    """
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    landlord_up_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_up'], 17)
    landlord_up_num_cards_left_batch = np.repeat(
        landlord_up_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_down_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_down'], 17)
    landlord_down_num_cards_left_batch = np.repeat(
        landlord_down_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_up_played_cards = _cards2array(
        infoset.played_cards['landlord_up'])
    landlord_up_played_cards_batch = np.repeat(
        landlord_up_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_down_played_cards = _cards2array(
        infoset.played_cards['landlord_down'])
    landlord_down_played_cards_batch = np.repeat(
        landlord_down_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         last_action_batch,
                         landlord_up_played_cards_batch,
                         landlord_down_played_cards_batch,
                         landlord_up_num_cards_left_batch,
                         landlord_down_num_cards_left_batch,
                         bomb_num_batch,
                         my_action_batch))
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             last_action,
                             landlord_up_played_cards,
                             landlord_down_played_cards,
                             landlord_up_num_cards_left,
                             landlord_down_num_cards_left,
                             bomb_num))
    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    obs = {
            'position': 'landlord',
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions,
            'x_no_action': x_no_action.astype(np.int8),
            'z': z.astype(np.int8),
          }
    return obs

def _get_obs_landlord_up(infoset):
    """
    Obttain the landlord_up features. See Table 5 in
    https://arxiv.org/pdf/2106.06135.pdf
    """
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    last_landlord_action = _cards2array(
        infoset.last_move_dict['landlord'])
    last_landlord_action_batch = np.repeat(
        last_landlord_action[np.newaxis, :],
        num_legal_actions, axis=0)
    landlord_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord'], 20)
    landlord_num_cards_left_batch = np.repeat(
        landlord_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    last_teammate_action = _cards2array(
        infoset.last_move_dict['landlord_down'])
    last_teammate_action_batch = np.repeat(
        last_teammate_action[np.newaxis, :],
        num_legal_actions, axis=0)
    teammate_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_down'], 17)
    teammate_num_cards_left_batch = np.repeat(
        teammate_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    teammate_played_cards = _cards2array(
        infoset.played_cards['landlord_down'])
    teammate_played_cards_batch = np.repeat(
        teammate_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         landlord_played_cards_batch,
                         teammate_played_cards_batch,
                         last_action_batch,
                         last_landlord_action_batch,
                         last_teammate_action_batch,
                         landlord_num_cards_left_batch,
                         teammate_num_cards_left_batch,
                         bomb_num_batch,
                         my_action_batch))
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             landlord_played_cards,
                             teammate_played_cards,
                             last_action,
                             last_landlord_action,
                             last_teammate_action,
                             landlord_num_cards_left,
                             teammate_num_cards_left,
                             bomb_num))
    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    obs = {
            'position': 'landlord_up',
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions,
            'x_no_action': x_no_action.astype(np.int8),
            'z': z.astype(np.int8),
          }
    return obs

def _get_obs_landlord_down(infoset):
    """
    Obttain the landlord_down features. See Table 5 in
    https://arxiv.org/pdf/2106.06135.pdf
    """
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    last_landlord_action = _cards2array(
        infoset.last_move_dict['landlord'])
    last_landlord_action_batch = np.repeat(
        last_landlord_action[np.newaxis, :],
        num_legal_actions, axis=0)
    landlord_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord'], 20)
    landlord_num_cards_left_batch = np.repeat(
        landlord_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    last_teammate_action = _cards2array(
        infoset.last_move_dict['landlord_up'])
    last_teammate_action_batch = np.repeat(
        last_teammate_action[np.newaxis, :],
        num_legal_actions, axis=0)
    teammate_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_up'], 17)
    teammate_num_cards_left_batch = np.repeat(
        teammate_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    teammate_played_cards = _cards2array(
        infoset.played_cards['landlord_up'])
    teammate_played_cards_batch = np.repeat(
        teammate_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         landlord_played_cards_batch,
                         teammate_played_cards_batch,
                         last_action_batch,
                         last_landlord_action_batch,
                         last_teammate_action_batch,
                         landlord_num_cards_left_batch,
                         teammate_num_cards_left_batch,
                         bomb_num_batch,
                         my_action_batch))
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             landlord_played_cards,
                             teammate_played_cards,
                             last_action,
                             last_landlord_action,
                             last_teammate_action,
                             landlord_num_cards_left,
                             teammate_num_cards_left,
                             bomb_num))
    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    obs = {
            'position': 'landlord_down',
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions,
            'x_no_action': x_no_action.astype(np.int8),
            'z': z.astype(np.int8),
          }
    return obs


# Public observations and QOJ score features

ENV_RANKS = tuple(range(3, 15))+(17, 20, 30)
ROLES = ('landlord', 'landlord_down', 'landlord_up')
CONTEXT_DIM, EXTRA_DIM = 48, 120


def role_for(seat, landlord):
    return ROLES[(seat-landlord)%3]


def reward_scale(role):
    return 6.0 if role == 'landlord' else 3.0


def public_snapshot(s):
    keys = ('id', 'version', 'phase', 'seat', 'turn', 'hand', 'players', 'landlord',
            'bottom', 'bid', 'multiplier', 'bombs', 'redeals', 'must_bid', 'leading', 'last', 'log')
    out = {k: copy.deepcopy(s[k]) for k in keys if k in s}
    out['players'] = [{k: p[k] for k in ('count', 'bid', 'landlord', 'auto') if k in p}
                      for p in out.get('players', [])]
    out['log'] = [{k: e[k] for k in ('kind', 'seat', 'cards', 'pattern', 'value') if k in e}
                  for e in out.get('log', [])]
    return out


def context(s):
    me = s['seat']; order = [me, (me+1)%3, (me+2)%3]
    logs = current_log(s)
    pc = Counter(e.get('seat') for e in logs if e.get('kind') == 'play')
    x = [float(s.get('bid', 0))/3, float(s.get('multiplier', 1))/16,
         float(s.get('bombs', 0))/14, min(10, s.get('redeals', 0))/10,
         float(bool(s.get('must_bid'))), float(bool(s.get('leading'))),
         float(s['phase'] == 'bidding')]
    x += [float(s.get('turn') == i) for i in order]
    players = s.get('players', [])
    x += [players[i].get('count', 0)/20 for i in order]
    # QOJ uses null for an unset/hidden bid; the local game uses -1.
    # Once playing starts, recover past bids from the public log as well.
    logged_bids = {e['seat']: e['value'] for e in logs if e.get('kind') == 'bid'}
    x += [(players[i]['bid'] if players[i].get('bid') is not None
           else logged_bids.get(i, -1))/3 for i in order]
    x += [pc[i]/20 for i in order]
    x += [float(s.get('landlord') == i) for i in order]
    x += [sum(e.get('kind') == 'bid' for e in logs)/3]
    x += [float((s.get('last') or {}).get('seat') == i) for i in order]
    land = s.get('landlord')
    x += [float(land is not None and all(pc[i] == 0 for i in range(3) if i != land)),
          float(land is not None and pc[land] <= 1)]
    return np.asarray(x+[0.]*(CONTEXT_DIM-len(x)), dtype=np.float32)


def prepare(s, actions):
    c = context(s)
    n = len(actions)
    if s['phase'] == 'bidding':
        hand = np.zeros((4, 15), dtype=np.float32)
        for r, count in Counter(map(rank, s['hand'])).items(): hand[:count, r] = 1
        rows = []
        for a in actions:
            one = np.zeros(4, dtype=np.float32); one[a.value] = 1
            rows.append(np.concatenate((hand.flatten(), c, one)))
        return 'bid', np.zeros((1, 5, 162), np.int8), np.stack(rows), np.zeros((n, EXTRA_DIM), np.float32)
    me, land = s['seat'], s['landlord']
    role = role_for(me, land)
    played = {r: [] for r in ROLES}; last = {r: [] for r in ROLES}; seq = []
    public_cards = []
    for e in current_log(s):
        if e.get('kind') not in ('play', 'pass'): continue
        cards = e.get('cards', []) if e['kind'] == 'play' else []
        encoded = [ENV_RANKS[rank(i)] for i in cards]
        seq.append(encoded)
        r = role_for(e['seat'], land)
        last[r] = encoded
        if e['kind'] == 'play':
            played[r].extend(encoded); public_cards.extend(cards)
    unseen = set(range(54))-set(s['hand'])-set(public_cards)
    lastcards = []
    if not s.get('leading'):
        target = s.get('last') or {}
        lastcards = target.get('cards')
        if lastcards is None:
            # The QOJ API supplies last={seat, pattern}; its cards are in log.
            entry = next((e for e in reversed(current_log(s)) if e.get('kind') == 'play'), None)
            if (entry is None or entry.get('seat') != target.get('seat')
                    or entry.get('pattern') != target.get('pattern')):
                raise ValueError('无法从公开出牌日志恢复待跟的牌，请 /refresh 同步状态。')
            lastcards = entry['cards']
    obs = get_obs(SimpleNamespace(
        player_position=role, player_hand_cards=[ENV_RANKS[rank(i)] for i in s['hand']],
        other_hand_cards=[ENV_RANKS[rank(i)] for i in sorted(unseen)],
        last_move=[ENV_RANKS[rank(i)] for i in lastcards], last_move_dict=last,
        played_cards=played, card_play_action_seq=seq, bomb_num=min(s.get('bombs', 0), 14),
        num_cards_left_dict={role_for(i, land): p['count'] for i, p in enumerate(s['players'])},
        legal_actions=[[ENV_RANKS[r] for r in a.ranks] for a in actions]))
    extras = []
    for a in actions:
        one = np.zeros(len(TYPE_ORDER), dtype=np.float32)
        if a.pattern: one[TYPE_ORDER.index(a.pattern[0])] = 1
        values = [float(a.kind == 'pass'), (a.pattern[1]/14 if a.pattern else 0),
                  (a.pattern[2]/12 if a.pattern else 0), len(a.ranks)/20]
        extras.append(np.concatenate((c, _cards2array([ENV_RANKS[r] for r in a.ranks]), one, values)))
    return role, obs['z'][None], obs['x_batch'], np.asarray(extras, dtype=np.float32)


# DouZero LSTM networks (Apache-2.0)

# Vendored upstream definitions; see README.md and LICENSE.txt.


class LandlordLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(373 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action)

class FarmerLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(484 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action)


# AlphaDou Net2 by Vincentzyx (GPL-3.0)

# Vendored upstream definitions; see README.md and LICENSE.txt.


class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        # input: 1 * 60
        self.conv1 = nn.Conv1d(1, 16, kernel_size=(3,), padding=1)  # 32 * 60
        self.dense1 = nn.Linear(1020, 1024)
        self.dense2 = nn.Linear(1024, 512)
        self.dense3 = nn.Linear(512, 256)
        self.dense4 = nn.Linear(256, 128)
        self.dense5 = nn.Linear(128, 1)

    def forward(self, xi):
        x = xi.unsqueeze(1)
        x = F.leaky_relu(self.conv1(x))
        x = x.flatten(1, 2)
        x = torch.cat((x, xi), 1)
        x = F.leaky_relu(self.dense1(x))
        x = F.leaky_relu(self.dense2(x))
        x = F.leaky_relu(self.dense3(x))
        x = F.leaky_relu(self.dense4(x))
        x = self.dense5(x)
        return x


# Trainable QOJ reward correction and bidding policy

class PlayModel(nn.Module):
    def __init__(self, landlord=False):
        super().__init__()
        self.base = LandlordLstmModel() if landlord else FarmerLstmModel()
        self.residual = nn.Sequential(nn.Linear(EXTRA_DIM, 128), nn.ReLU(), nn.Linear(128, 1))
        nn.init.zeros_(self.residual[-1].weight); nn.init.zeros_(self.residual[-1].bias)

    def dense(self, h, x, extra):
        y = torch.cat((h, x), dim=-1)
        for i in range(1, 6): y = torch.relu(getattr(self.base, f'dense{i}')(y))
        return self.base.dense6(y)+self.residual(extra)

    def forward(self, z, x, extra):
        h = self.base.lstm(z)[0][:, -1, :]
        return self.dense(h, x, extra)


class BidModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.strength = Net2()
        self.strength.requires_grad_(False)
        self.residual = nn.Sequential(nn.Linear(112, 128), nn.ReLU(), nn.Linear(128, 128),
                                      nn.ReLU(), nn.Linear(128, 1))
        nn.init.zeros_(self.residual[-1].weight); nn.init.zeros_(self.residual[-1].bias)

    def forward(self, z, x, extra):
        with torch.no_grad():
            score = self.strength(x[:, :60]).squeeze(-1)
            preferred = (score > -0.1).float()+(score > 0).float()+(score > 0.1).float()
            actual = x[:, -4:].argmax(dim=1).float()
            prior = -0.1*torch.abs(actual-preferred)
        return prior[:, None]+self.residual(x)


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.models = nn.ModuleDict({**{r: PlayModel(r == 'landlord') for r in ROLES}, 'bid': BidModel()})

    def forward(self, role, z, x, extra):
        return self.models[role](z, x, extra)

    def optimizer_groups(self, base_lr=1e-5, new_lr=1e-4):
        base, new = [], []
        for r in ROLES:
            base.extend(self.models[r].base.parameters()); new.extend(self.models[r].residual.parameters())
        new.extend(self.models['bid'].residual.parameters())
        return [{'params': base, 'lr': base_lr}, {'params': new, 'lr': new_lr}]


# CPU model bundles and atomic persistence

SCHEMA = 1


def atomic_save(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.writing-', suffix='.pt', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            torch.save(value, f); f.flush(); os.fsync(f.fileno())
        for attempt in range(100):
            try: os.replace(name, path); break
            except PermissionError:
                if attempt == 99: raise
                time.sleep(0.1)
    finally:
        if os.path.exists(name): os.unlink(name)


def bundle(policy, meta):
    return {'schema': SCHEMA, 'weights': {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()},
            'meta': dict(meta)}


def load_bundle(path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data.get('schema') != SCHEMA: raise ValueError('模型格式不兼容。')
    policy = Policy(); policy.load_state_dict(data['weights'], strict=True); policy.eval()
    return policy, data


# Model recommendation and hot reload

class Agent:
    def __init__(self, path, threads=2):
        torch.set_num_threads(threads)
        self.path = Path(path)
        # Stat before reading: a concurrent publication must trigger a reload,
        # never mark old weights as having the new file's signature.
        self.signature = self._signature()
        self.policy, data = load_bundle(self.path)
        self.meta = data['meta']

    def _signature(self):
        s = self.path.stat(); return s.st_mtime_ns, s.st_size

    def reload(self):
        sig = self._signature()
        if sig != self.signature:
            candidate, data = load_bundle(self.path)
            self.policy, self.meta, self.signature = candidate, data['meta'], sig

    @torch.inference_mode()
    def values(self, s, actions, return_features=False):
        role, z, x, extra = prepare(s, actions)
        model = self.policy.models[role]
        zt = torch.from_numpy(z.astype(np.float32))
        if role != 'bid': h = model.base.lstm(zt)[0][:, -1, :]
        values = []
        for start in range(0, len(actions), 256):
            xt = torch.from_numpy(x[start:start+256].astype(np.float32))
            et = torch.from_numpy(extra[start:start+256])
            y = model(zt, xt, et) if role == 'bid' else model.dense(h.expand(len(xt), -1), xt, et)
            values.append(y.squeeze(-1).numpy())
        result = np.concatenate(values)
        return (result, (role, z, x, extra)) if return_features else result

    def recommend(self, s, top=5):
        self.reload()
        actions = legal_actions(s)
        if not actions: raise ValueError('没有合法动作。')
        values = self.values(s, actions)
        role = 'bid' if s['phase'] == 'bidding' else role_for(s['seat'], s['landlord'])
        order = np.argsort(-values, kind='stable')[:top]
        return {'action': actions[int(order[0])].to_dict(),
                'suggestions': [{'action': actions[int(i)].to_dict(), 'value': float(values[i])*reward_scale(role)} for i in order],
                'fallback': False, 'model': {'frames': self.meta.get('frames', 0), 'updates': self.meta.get('updates', 0)}}


# Bounded inference worker and legal fallback

def _worker(pipe, path, threads):
    try:
        agent = Agent(path, threads)
        pipe.send({'ready': True})
        while True:
            request = pipe.recv()
            if request is None: break
            started = time.monotonic()
            try: result = agent.recommend(request['state'])
            except Exception as e: result = {'error': f'{type(e).__name__}: {e}'}
            pipe.send({'request_id': request['request_id'], 'result': result,
                       'compute_elapsed': time.monotonic()-started})
    except (EOFError, BrokenPipeError): pass
    except Exception as e:
        try: pipe.send({'error': f'{type(e).__name__}: {e}'})
        except (BrokenPipeError, EOFError, OSError): pass
    finally: pipe.close()


class BoundedEngine:
    def __init__(self, path, threads=2, budget=4.0):
        self.path, self.threads, self.budget = str(path), threads, budget
        self.lock = threading.Lock(); self.process = self.pipe = None; self.ready = False
        self.pending = None; self.sequence = 0
        self._start()

    def _start(self):
        parent, child = mp.get_context('spawn').Pipe()
        self.process = mp.get_context('spawn').Process(target=_worker, args=(child, self.path, self.threads), daemon=True)
        self.process.start(); child.close(); self.pipe = parent; self.ready = False; self.pending = None

    def _stop(self):
        if self.process is not None:
            if self.process.is_alive(): self.process.terminate()
            self.process.join(timeout=0.1)
        if self.pipe is not None: self.pipe.close()
        self.process = self.pipe = None; self.ready = False; self.pending = None

    def choose(self, state):
        start = time.monotonic(); deadline = start+self.budget
        result = {'action': minimal_action(state).to_dict(), 'suggestions': [], 'fallback': True,
                  'retry': True, 'status': 'busy', 'reason': '本地引擎正在处理上一条请求。'}
        acquired = self.lock.acquire(timeout=max(0, deadline-time.monotonic()))
        try:
            if acquired:
                snapshot = public_snapshot(state)
                if self.process is None: self._start()
                if not self.process.is_alive() and not self.pipe.poll():
                    raise OSError(f'模型进程退出（exitcode={self.process.exitcode}）。')
                if not self.ready:
                    result.update(status='loading', reason='模型正在加载，尚未生成决策。')
                    if not self.pipe.poll(max(0, deadline-time.monotonic())): return result
                    message = self.pipe.recv()
                    if not message.get('ready'):
                        result.update(retry=False, status='error', reason=message.get('error', '模型初始化失败。'))
                        self._stop(); return result
                    self.ready = True
                # A short wait after cold loading must not kill the now-warm model.
                # Keep at most one request, and correlate late replies with its exact
                # public position before returning them to either UI.
                while True:
                    if self.pending is None:
                        if time.monotonic() >= deadline:
                            result.update(status='busy', reason='模型已加载，等待下一次决策。')
                            return result
                        self.sequence += 1
                        sent = time.monotonic()
                        self.pending = {'request_id': self.sequence, 'state': snapshot,
                                        'deadline': sent+self.budget, 'budget': self.budget}
                        self.pipe.send({'request_id': self.sequence, 'state': snapshot})
                    pending = self.pending
                    remaining = min(deadline, pending['deadline'])-time.monotonic()
                    if not self.pipe.poll(max(0, remaining)):
                        if time.monotonic() >= pending['deadline']:
                            result.update(retry=False, status='timeout', reason='模型计算超过决策时限；托管暂停，请检查 CPU 占用。')
                            self._stop()
                        else:
                            result.update(status='busy', reason='模型已加载，正在计算；尚未生成决策。')
                        return result
                    message = self.pipe.recv()
                    if message.get('request_id') != pending['request_id']:
                        raise ValueError('模型回复与请求编号不一致。')
                    self.pending = None
                    # Never apply a late answer to a new hand, turn, or game.
                    if pending['state'] != snapshot: continue
                    answer = message['result']
                    if 'error' in answer:
                        result.update(retry=False, status='error', reason=answer['error'])
                    elif message['compute_elapsed'] > pending['budget']:
                        result.update(retry=False, status='timeout', reason='模型计算超过决策时限；托管暂停，请检查 CPU 占用。')
                    else:
                        result = answer
                        result.update(retry=False, status='ready')
                    return result
        except (EOFError, BrokenPipeError, OSError, ValueError) as e:
            result.update(retry=False, status='error', reason=f'本地引擎错误：{e}'); self._stop()
        finally:
            if acquired: self.lock.release()
            result['elapsed'] = time.monotonic()-start
        return result

    def close(self):
        with self.lock: self._stop()


# Nine-round score match

VERSION = "2.0.0"
INTERMISSION_SECONDS = 5.0
MODES = {"single": "单局匹配", "match": "记分比赛"}


def text(value):
    return "".join(c if unicodedata.category(c) not in ("Cc", "Cf", "Cs") else " "
                   for c in str(value))


def at(values, index, default=None):
    if isinstance(values, (list, tuple)) and 0 <= index < len(values):
        return values[index]
    return default


def signed(value):
    return f"{value:+}" if isinstance(value, (int, float)) else "—"


def width(value):
    try:
        from wcwidth import wcswidth
        return max(0, wcswidth(value))
    except ImportError:
        return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in value)


def fitted(value, size, right=False):
    value = text(value)
    if width(value) > size:
        short = ""
        for char in value:
            if width(short + char + "…") > size:
                break
            short += char
        value = short + "…"
    padding = " " * max(0, size - width(value))
    return padding + value if right else value + padding


def match_of(state):
    return (state or {}).get("match") or {}


def between_rounds(state):
    return bool(state and state.get("phase") == "finished" and match_of(state)
                and not match_of(state).get("finished"))


def match_player_order(state):
    """Map seat order by username: match arrays are not assumed to use seat IDs."""
    match = match_of(state)
    names = match.get("players") or []
    seat = state.get("seat")
    players = state.get("players") or []
    indices = []
    if isinstance(seat, int) and len(players) == 3:
        for i in ((seat + 1) % 3, (seat + 2) % 3, seat):
            username = players[i].get("username")
            if username in names:
                index = names.index(username)
                if index not in indices:
                    indices.append(index)
    return indices + [i for i in range(len(names)) if i not in indices]


def own_name(state):
    player = at((state or {}).get("players"), (state or {}).get("seat", -1), {}) if isinstance((state or {}).get("seat"), int) else {}
    return player.get("username", "")


def match_summary(state):
    m = match_of(state)
    names = m.get("players") or []
    return "累计分：" + "  ".join(
        f"{text(names[i])}{' [我]' if names[i] == own_name(state) else ''} {signed(at(m.get('totals'), i))}"
        for i in match_player_order(state))


def scoreboard_lines(state, columns=100):
    m = match_of(state)
    if not m:
        return ["本局是单局匹配，没有九局比赛分表。"]
    names = m.get("players") or []
    rounds = int(m.get("rounds") or 9)
    finished = bool(m.get("finished"))
    title = f"记分比赛 #{m.get('id', '—')} · "
    title += "比赛结束" if finished else f"第 {m.get('round', '—')}/{rounds} 局"
    rows = [title, ""]
    deltas = m.get("deltas") or []
    totals = m.get("totals") or []
    me = own_name(state)
    labels = [text(name) + (" [我]" if name == me else "") for name in names]
    order = match_player_order(state)
    name_size = max(10, min(26, max((width(s) for s in labels), default=10)))
    numbers = [signed(at(delta, i)) for delta in deltas for i in range(len(names))]
    cell_size = max(3, max((width(s) for s in numbers), default=3))
    total_size = max(4, max((width(signed(v)) for v in totals), default=4))
    # Split columns on small terminals instead of dropping later rounds or totals.
    per_block = max(1, min(rounds, (max(40, columns) - name_size - total_size - 4) // (cell_size + 1)))
    for start in range(0, rounds, per_block):
        stop = min(rounds, start + per_block)
        heading = fitted("玩家 / 局", name_size) + "  "
        heading += " ".join(fitted(str(r + 1), cell_size, True) for r in range(start, stop))
        rows.append(heading + "  " + fitted("总分", total_size, True))
        for i in order:
            values = [signed(at(at(deltas, r, []), i)) for r in range(start, stop)]
            row = fitted(labels[i], name_size) + "  "
            row += " ".join(fitted(v, cell_size, True) for v in values)
            rows.append(row + "  " + fitted(signed(at(totals, i)), total_size, True))
        rows.append("")
    if any(width(label) > name_size for label in labels):
        rows.extend("玩家：" + label for label in labels)
        rows.append("")
    if finished:
        rows.append("最终名次 / Rating")
        for i in order:
            place = at(m.get("places"), i)
            before, after = at(m.get("rating_before"), i), at(m.get("rating_after"), i)
            ranking = f"#{place}" if place is not None else "名次待同步"
            if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                rating = f"Rating {before} → {after} ({signed(after - before)})"
            elif after is not None:
                rating = f"Rating {after}"
            else:
                rating = "Rating 待服务器同步"
            rows.append(f"{ranking}  {labels[i]}  总分 {signed(at(totals, i))}  {rating}")
    else:
        rows.append("名次和 Rating 由整场比赛结束后的服务器结果决定。")
    result = state.get("result")
    if result:
        tags = (" · 春天" if result.get("spring") else "") + (" · 反春" if result.get("anti_spring") else "")
        rows += ["", f"本局：{'地主胜' if result.get('landlord_won') else '农民胜'}{tags} · 底分 {result.get('base', '—')} × {result.get('multiplier', '—')}"]
    return rows


@dataclass(frozen=True)
class Submitted:
    line: str
    game_id: int | None
    during_break: bool = False


class MatchInputQueue(asyncio.Queue):
    def __init__(self, client):
        super().__init__(maxsize=1)
        self.client = client

    def put_nowait(self, line):
        state = self.client.state
        if isinstance(line, str) and (not state or match_of(state)):
            line = Submitted(line, (state or {}).get("id"), between_rounds(state))
        return super().put_nowait(line)


def _install_qoj_match(ns):
    """Install once in the original module's namespace, before its main() call."""
    if ns.get("_QOJ_MATCH_INSTALLED"):
        return
    required = ("QojClient", "TerminalUI", "ClientError", "LoginError", "ApiError", "parse_game", "soup_of", "json", "LOBBY", "HELP")
    missing = [name for name in required if name not in ns]
    if missing:
        raise RuntimeError("qoj_cli.py 版本不兼容，缺少：" + ", ".join(missing))
    BaseClient, BaseUI = ns["QojClient"], ns["TerminalUI"]
    ClientError, LoginError, ApiError = ns["ClientError"], ns["LoginError"], ns["ApiError"]
    original_parse_game = ns["parse_game"]

    def parse_game(html):
        # Preserve every original single-game check; only remove the match rejection.
        soup = ns["soup_of"](html)
        root, initial = soup.select_one("#ddz"), soup.select_one("#ddz-initial")
        if root is None or initial is None:
            return original_parse_game(html)
        data = ns["json"].loads(initial.get_text())
        if not data.get("state", {}).get("match"):
            return original_parse_game(html)
        if root.get("data-logged-in") != "1" or data["state"].get("seat") is None:
            raise LoginError("该 Cookie 不是这场比赛的参赛者，不能出牌。")
        return root.get("data-token", ""), data

    class MatchClient(BaseClient):
        def __init__(self, *args, **kwargs):
            self.queue_mode = "single"
            super().__init__(*args, **kwargs)

        def accept(self, data):
            super().accept(data)
            update = data.get("state")
            if isinstance(update, dict) and update.get("unchanged") and isinstance(update.get("match"), dict) and self.state:
                # Some servers attach refreshed match metadata without a new move version.
                incoming = update["match"]
                if incoming.get("id") == match_of(self.state).get("id"):
                    self.state = {**self.state, "match": incoming}
            if self.state:
                self.queue_mode = "match" if match_of(self.state) else "single"
            elif self.lobby_status.get("queued") in MODES:
                self.queue_mode = self.lobby_status["queued"]

        async def call(self, action, **extra):
            if action == "queue":
                mode = extra.get("mode", self.queue_mode)
                if mode not in MODES:
                    raise ClientError("匹配模式必须是 single 或 match。")
                self.queue_mode = mode
                extra["mode"] = mode
            elif action in ("lobby", "cancel"):
                extra.setdefault("mode", self.queue_mode)
            return await super().call(action, **extra)

        async def load_game(self, game_id, *, expected_match=None, after_round=None):
            if expected_match is None:
                await super().load_game(game_id)
                return
            # Validate next game's identity before discarding the current score table.
            html = await self.get_html(f"{ns['LOBBY']}/game/{int(game_id)}?locale=zh-cn")
            token, data = parse_game(html)
            state = data["state"]
            m = match_of(state)
            if (state.get("id") != int(game_id) or m.get("id") != expected_match
                    or int(m.get("round") or 0) <= int(after_round or 0)):
                raise ClientError("下一局尚未准备好，继续显示分表并等待同步。")
            if own_name(state) != own_name(self.state):
                raise ClientError("下一局玩家身份不一致，已保留当前分表。")
            self.token = token
            self.state = None
            self.chat, self.chat_after, self.commitments = {}, 0, {}
            self.accept(data)

    class MatchUI(BaseUI):
        def __init__(self, *args, **kwargs):
            self._break_game = None
            self._break_deadline = 0.0
            self._clock = time.monotonic
            super().__init__(*args, **kwargs)
            self.commands = MatchInputQueue(self.client)
            self._install_score_screen()

        def _install_score_screen(self):
            from prompt_toolkit.layout import HSplit, VSplit, Layout, Window, DynamicContainer
            from prompt_toolkit.layout.containers import ConditionalContainer
            from prompt_toolkit.layout.controls import FormattedTextControl
            from prompt_toolkit.filters import Condition
            old_root = self.app.layout.container
            self.score_window = Window(FormattedTextControl(self.render_score_screen), wrap_lines=True)
            score_root = HSplit([
                Window(FormattedTextControl(self.render_header), height=1, style="class:header"),
                Window(FormattedTextControl(" 比赛分表"), height=1, style="class:section"),
                self.score_window,
                ConditionalContainer(HSplit([
                    Window(FormattedTextControl(" 聊天（/chat 查看全部）"), height=1, style="class:section"),
                    Window(FormattedTextControl(self.render_chat), height=3, wrap_lines=True),
                ]), Condition(lambda: bool(match_of(self.client.state).get("finished")))),
                Window(FormattedTextControl(lambda: text(self.notice)), height=2, wrap_lines=True, style="class:notice"),
                Window(FormattedTextControl(self.render_prompt), height=2, wrap_lines=True),
                VSplit([Window(FormattedTextControl("> "), width=2, height=1),
                        Window(self.input_control, height=1)], style="class:input"),
            ])

            def current_root():
                s = self.client.state
                if between_rounds(s):
                    return score_root
                if s and s.get("phase") == "finished" and match_of(s).get("finished") and not self.detail:
                    return score_root
                return old_root
            self.app.layout = Layout(DynamicContainer(current_root), self.input_control)

        def render_header(self):
            s = self.client.state
            m = match_of(s)
            if m:
                return super().render_header().replace("单局匹配", f"记分比赛 #{m['id']} · {m.get('round', '—')}/{m.get('rounds', 9)} 局", 1)
            return super().render_header().replace("单局匹配", "单局 / 记分比赛", 1)

        def render_home(self):
            output = super().render_home().replace("2  记分比赛（后续实现）", "2  记分比赛（同三人连续 9 局）")
            status = self.client.lobby_status
            if status.get("queued") == "match":
                output = output.replace(
                    f"同桌积分差 ≤ {status.get('window', '—')} · 我的积分 {status.get('score', '—')}",
                    f"同桌 Rating 差 ≤ {status.get('window', '—')} · 我的 Rating {status.get('rating', self.client.lobby.rating)}")
            return output

        def render_board(self):
            board = super().render_board()
            if match_of(self.client.state):
                return match_summary(self.client.state) + "  (/score 分表)\n" + board
            return board

        def render_score_screen(self):
            s = self.client.state
            rows = scoreboard_lines(s, self.app.output.get_size().columns)
            if between_rounds(s):
                rows += ["", "局间等待：准备好后自动进入下一局。"]
            else:
                rows += ["", "整场比赛已结束。/again 再来一场；/home 回大厅；/score 查看完整分表。"]
            return "\n".join(rows)

        def render_prompt(self):
            s = self.client.state
            if not s:
                return "输入 1 单局匹配；2 记分比赛（9 局）；/help 操作说明。"
            if between_rounds(s):
                return "等待下一局自动开始；/refresh 同步；/quit 退出。"
            if match_of(s).get("finished") and s.get("phase") == "finished":
                return "比赛结束：/again 再来一场；/home 大厅；普通文字仍可聊天。"
            return super().render_prompt()

        async def enter_or_queue(self, mode=None):
            if mode is not None:
                self.client.queue_mode = mode
            await super().enter_or_queue()
            if self.client.lobby_status.get("queued") == "match" and not self.client.state:
                self.set_notice("已加入记分比赛匹配，等待三人到齐。")
            self.observe_match()

        def observe_match(self):
            s = self.client.state
            if between_rounds(s):
                key = (match_of(s).get("id"), s.get("id"))
                if self._break_game != key:
                    self._break_game = key
                    self._break_deadline = self._clock() + INTERMISSION_SECONDS
                    self.pending = None
                    self.detail = ""
                    self.buffer.reset()
                    self.set_notice("本局结束，显示累计分表；下一局将自动开始。")
            else:
                self._break_game = None
                self._break_deadline = 0.0

        async def advance_match(self):
            """Only GET the server-issued next game. Never queue/create another game."""
            self.observe_match()
            s = self.client.state
            if not between_rounds(s):
                return False
            m = match_of(s)
            next_game = m.get("next_game")
            if not next_game or self._clock() < self._break_deadline:
                return False
            if not isinstance(next_game, int) or next_game <= 0 or next_game == s.get("id"):
                raise ClientError("服务器返回的下一局编号无效，正在等待重新同步。")
            await self.client.load_game(next_game, expected_match=m["id"], after_round=m.get("round", 0))
            self.pending, self.detail = None, ""
            self.scroll = {"log": 0, "chat": 0}
            self.buffer.reset()  # Old-round cards/chat must never be sent into the next hand.
            self.observe_match()
            new = match_of(self.client.state)
            self.set_notice(f"已进入记分比赛第 {new.get('round', '—')}/{new.get('rounds', 9)} 局。")
            self.app.invalidate()
            return True

        async def command(self, line):
            stamp = line if isinstance(line, Submitted) else None
            line = stamp.line if stamp else line
            cmd = line.partition(" ")[0].lower()
            s = self.client.state
            m = match_of(s)
            safe_local = ("/quit", "/exit", "/help", "/?", "/close", "/score", "/scores", "/refresh", "/browser")
            if stamp and cmd not in safe_local and (stamp.game_id != (s or {}).get("id") or stamp.during_break):
                raise ClientError("这条输入来自上一局或局间等待，未提交；请根据当前手牌重新输入。")
            if cmd in ("/score", "/scores"):
                if not m:
                    raise ClientError("当前没有记分比赛。")
                if not between_rounds(s):
                    self.show_detail("比赛分表", scoreboard_lines(s, self.app.output.get_size().columns))
                return
            if m and not m.get("finished") and cmd in ("/home", "/again"):
                raise ClientError("整场比赛尚未结束；局间会自动换局。")
            if between_rounds(s) and cmd not in safe_local:
                raise ClientError("正在局间等待，下一局会自动开始。")
            if not s and line in ("1", "2") and self.connected and not self.demo:
                await self.enter_or_queue("single" if line == "1" else "match")
                return
            if cmd == "/again" and m and m.get("finished"):
                self.client.queue_mode = "match"
            await super().command(line)
            self.observe_match()

        async def connect(self):
            await super().connect()
            self.observe_match()

        async def poll_once(self):
            was_queued = self.client.queue_may_be_active or bool(self.client.lobby_status.get("queued"))
            await self.client.call("state" if self.client.state else "lobby")
            if not self.client.state and was_queued and self.client.lobby_status.get("game"):
                await self.client.load_game(self.client.lobby_status["game"])
                self.set_notice("匹配成功，已进入对局。")
            self.observe_match()
            await self.advance_match()
            # Keep an explicitly opened score detail live while a game is in progress.
            if self.detail == "比赛分表" and match_of(self.client.state):
                lines = scoreboard_lines(self.client.state, self.app.output.get_size().columns)
                if "\n".join(lines) not in self.details.buffer.text:
                    self.show_detail("比赛分表", lines)

        async def poll_loop(self):
            failures = 0
            while True:
                state = self.client.state
                delay = 0.9 if state and state.get("phase") != "finished" else 3.0
                if between_rounds(state):
                    delay = 0.9
                elif not state:
                    delay = 1.5 if self.client.lobby_status.get("queued") else 5.0
                if failures:
                    delay = min(10, 1.5 * 2 ** min(failures, 4))
                self.wakeup.clear()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self.wakeup.wait(), timeout=delay)
                if not self.connected or self.busy:
                    continue
                try:
                    async with self.op_lock:
                        await self.poll_once()
                        if failures:
                            self.set_notice("连接已恢复；已同步最新状态。")
                        failures = 0
                        self.app.invalidate()
                except LoginError as exc:
                    self.connected = False
                    self.set_notice(str(exc))
                except ClientError as exc:
                    failures += 1
                    self.set_notice(f"状态同步失败：{exc}（会重试）")
                    if isinstance(exc, ApiError) and exc.retry_after:
                        await asyncio.sleep(exc.retry_after)
                except Exception as exc:
                    failures += 1
                    self.set_notice(f"状态格式发生变化（{type(exc).__name__}）；请提供最新比赛页面。")

    ns["parse_game"] = parse_game
    ns["QojClient"] = MatchClient
    ns["TerminalUI"] = MatchUI
    ns["HELP"] = ns["HELP"].replace("大厅：1 单局匹配，2 记分比赛（本版未实现，不会加入队列）。",
                                   "大厅：1 单局匹配，2 记分比赛（同三人连续 9 局）。")
    ns["HELP"] += ("\n/score 查看比赛每局得分、累计积分；终局显示名次和 Rating 变化。"
                   "\n记分比赛局间仅显示分表并自动进入下一局；整场结束后 /again 再来一场。")
    ns["_QOJ_MATCH_INSTALLED"] = VERSION


# Local AI hints and automatic play

MODEL_PATH = Path(__file__).resolve().parent/'models'/'current.pt'


def _key(s):
    return (s or {}).get('id'), (s or {}).get('version')


def _position(s):
    """Do not resolve an uncertain move merely because chat changed the version."""
    return ((s or {}).get('id'), (s or {}).get('phase'), (s or {}).get('turn'),
            tuple((s or {}).get('hand') or []), (s or {}).get('bid'),
            len([e for e in (s or {}).get('log', []) if e.get('kind') in ('bid', 'play', 'pass', 'redeal')]))


def _install_qoj_ai(ns):
    if ns.get('_QOJ_AI_INSTALLED'): return
    BaseUI, ClientError, NetworkError = ns['TerminalUI'], ns['ClientError'], ns['NetworkError']

    class AIUI(BaseUI):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.ai_engine = None; self.ai_bot = False
            self.ai_attempt = self.ai_uncertain = None

        def engine(self):
            if self.ai_engine is None: self.ai_engine = BoundedEngine(MODEL_PATH)
            return self.ai_engine

        async def command(self, line):
            # MatchInputQueue stamps all input. Keep its stale-round protection.
            stamp = line if hasattr(line, 'line') and hasattr(line, 'game_id') else None
            raw = stamp.line if stamp else line
            cmd, _, arg = raw.strip().partition(' '); cmd = cmd.lower(); arg = arg.strip().lower()
            s = self.client.state
            if cmd not in ('/ai', '/bot'):
                if cmd == '/auto' and arg in ('on', '1'): self.ai_bot = False
                try: return await super().command(line)
                except NetworkError:
                    self.ai_bot = False
                    if s and (cmd in ('/play', '/pass', '/choose', '/bid', '/auto') or raw == '-'
                              or (not cmd.startswith('/') and ns['parse_ranks'](raw) is not None)):
                        self.ai_uncertain = _position(s)
                    raise
            if cmd == '/bot' and arg == 'off':
                self.ai_bot = False; self.set_notice('本地模型托管已关闭。'); return
            if stamp and (stamp.game_id != (s or {}).get('id') or stamp.during_break):
                raise ClientError('这条输入来自上一局或局间等待，请重新输入。')
            if cmd == '/ai':
                if arg: raise ClientError('用法：/ai')
                if not s or s.get('phase') not in ('bidding', 'playing') or s.get('turn') != s.get('seat'):
                    raise ClientError('请在轮到你叫分或出牌时使用 /ai。')
                key = _key(s)
                result = await asyncio.to_thread(self.engine().choose, public_snapshot(s))
                if _key(self.client.state) != key:
                    self.set_notice('牌局已变化，旧提示已丢弃。'); return
                rows = []
                if result['fallback']: rows.append('未生成模型提示：'+result['reason'])
                else:
                    rows += [f"{i+1}. {action_text(Action.from_dict(row['action']))} · 估值 {row['value']:+.3f}"
                             for i, row in enumerate(result['suggestions'])]
                    rows.append('估值是模型分数，不是胜率；初始叫分估值尚未按 QOJ 分数校准。')
                rows.append(f"本地决策 {result['elapsed']:.3f}s；不会自动提交提示。")
                self.show_detail('本地模型提示', rows); return
            if arg != 'on': raise ClientError('用法：/bot on 或 /bot off')
            if self.demo: raise ClientError('离线演示不能提交动作；可使用 /ai。')
            if not self.connected: raise ClientError('尚未连接，请先 /refresh。')
            if not s or s.get('phase') not in ('bidding', 'playing'): raise ClientError('请在对局中开启本地托管。')
            if self.ai_uncertain == _position(s):
                raise ClientError('上次提交结果仍不确定，请 /refresh 并核对牌局，待出牌状态变化后再开启。')
            self.engine()
            if s['players'][s['seat']].get('auto'):
                await self.client.call('auto', on='0')
            self.ai_attempt = None; self.ai_bot = True
            self.set_notice('本地模型托管已开启；/bot off 关闭。')

        async def ai_step(self):
            s = self.client.state
            if not self.ai_bot or self.busy or not self.connected or not s: return
            if s.get('phase') not in ('bidding', 'playing'): return
            # A server timeout can re-enable its own bot between polls. Clear
            # that flag even on opponents' turns, before it can auto-pass again.
            if s['players'][s['seat']].get('auto'):
                try:
                    async with self.op_lock:
                        fresh = self.client.state
                        if (not self.ai_bot or not fresh
                                or fresh.get('phase') not in ('bidding', 'playing')): return
                        if fresh['players'][fresh['seat']].get('auto'):
                            await self.client.call('auto', on='0')
                            self.set_notice('已取消服务器托管，由本地模型继续接管。')
                        self.ai_attempt = None
                        self.wakeup.set()
                except Exception as e:
                    self.ai_bot = False
                    self.set_notice(f'取消服务器托管失败，本地托管已暂停：{e}')
                return
            if s.get('turn') != s.get('seat'): return
            key = _key(s)
            if key == self.ai_attempt: return
            self.ai_attempt = key
            result = await asyncio.to_thread(self.engine().choose, public_snapshot(s))
            if not self.ai_bot: return
            if self.busy:
                self.ai_attempt = None; return
            if result['fallback']:
                if _key(self.client.state) != key: return
                if result.get('retry'):
                    self.ai_attempt = None
                    self.set_notice('本地托管等待：'+result['reason'])
                else:
                    self.ai_bot = False
                    self.set_notice('本地托管已暂停：'+result['reason'])
                return
            submitted = False
            try:
                async with self.op_lock:
                    if not self.ai_bot or _key(self.client.state) != key: return
                    # State is read-only. Never replay a move after ambiguous network failure.
                    await self.client.call('state')
                    fresh = self.client.state
                    if _key(fresh) != key or fresh.get('turn') != fresh.get('seat'): return
                    if fresh['players'][fresh['seat']].get('auto'):
                        await self.client.call('auto', on='0'); self.ai_attempt = None; return
                    a = Action.from_dict(result['action'])
                    cards = validate_action(fresh, a)
                    submitted = True
                    if a.kind == 'bid': await self.client.call('bid', value=str(a.value))
                    elif a.kind == 'pass': await self.client.call('pass')
                    else:
                        await self.client.call('play', cards=','.join(map(str, cards)), choice=ns['pattern_key'](a.reading))
                    self.pending = None
                    self.set_notice('本地托管：'+action_text(a)+(('；'+result['reason']) if result['fallback'] else ''))
                    self.wakeup.set(); self.app.invalidate()
            except NetworkError as e:
                self.ai_bot = False
                if submitted: self.ai_uncertain = _position(s)
                self.set_notice(f'{e} 本地托管已暂停；结果不确定的动作不会重发。')
            except Exception as e:
                self.ai_bot = False; self.set_notice(f'本地托管已暂停：{e}')

        async def ai_loop(self):
            while True:
                try: await self.ai_step()
                except asyncio.CancelledError: raise
                except Exception as e:
                    self.ai_bot = False; self.set_notice(f'本地引擎已暂停：{e}')
                await asyncio.sleep(0.15)

        async def run(self):
            task = asyncio.create_task(self.ai_loop())
            try: await super().run()
            finally:
                self.ai_bot = False; task.cancel()
                with contextlib.suppress(asyncio.CancelledError): await task
                if self.ai_engine is not None: await asyncio.to_thread(self.ai_engine.close)

    ns['TerminalUI'] = AIUI
    ns['HELP'] += '\n/ai 本地模型提示（叫分、出牌）；/bot on|off 本地模型托管。\n/auto 仍是原有服务器托管；开启服务器托管会关闭本地托管。'
    ns['_QOJ_AI_INSTALLED'] = True


ROOT = Path(__file__).resolve().parents[1]


def package_client(output):
    client = Path(__file__).resolve().parent
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = [client/'qoj_cli.py', client/'requirements.txt',
             client/'LICENSE.txt', client/'models/current.pt']
    for file in files:
        if not file.is_file(): raise ClientError('客户端文件缺失：'+str(file))
    if output in files: raise ClientError('ZIP 输出不能覆盖客户端源码或模型。')
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for file in files:
            archive.write(file, Path('qoj_bot_client/client')/file.relative_to(client))
        readme = ROOT/'README.md'
        if readme.is_file(): archive.write(readme, 'qoj_bot_client/README.md')
    print('已打包客户端：'+str(output))


# Stable UI base for the local table; online extensions keep their own commands.
BaseTerminalUI = TerminalUI
_install_qoj_match(globals())
_install_qoj_ai(globals())

if __name__ == '__main__':
    mp.freeze_support()
    raise SystemExit(main())
