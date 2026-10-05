#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QOJ 斗地主：单局、九局比赛、本地 CPU 机器人和共享规则/模型。

在线：python client/qoj_cli.py
离线训练：python train.py；本地对战：python local.py
来源、许可、依赖和操作说明见唯一的 README.md；第三方许可见 LICENSE.txt。
final-v2：自行定义的全公开历史网络；不加载 DouZero / AlphaDou 网络或权重。
保留 final-v1 的 QOJ 协议、规则、界面与许可历史；模型格式升级为 schema 2。
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
import torch
from torch import nn
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
        self.bidder = None
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
            if a.value: self.bid, self.bidder = a.value, seat
            if a.value == 3 or (self.bid_count == 3 and self.bid):
                self.landlord = self.bidder
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


# Custom public-information model. No upstream network definitions or weights.

ROLES = ('landlord', 'landlord_down', 'landlord_up')
EVENT_KINDS = ('start', 'bid', 'redeal', 'landlord', 'play', 'pass', 'finish')
STATE_DIM, EVENT_DIM, ACTION_DIM = 264, 68, 57
SCHEMA = 2
ARCHITECTURE = {'name': 'QOJ-FullHistoryQ', 'version': 1, 'state': STATE_DIM,
                'event': EVENT_DIM, 'action': ACTION_DIM, 'gru': 128, 'hidden': 256}
SCORE_SCALE = 6.0  # One constant for ALL seats, including bidding.


def role_for(seat, landlord):
    return ROLES[(seat-landlord)%3]


def public_snapshot(s):
    """Whitelist only one's own hand and public facts, never fairness/hands."""
    keys = ('id', 'version', 'phase', 'seat', 'turn', 'hand', 'players', 'landlord',
            'bottom', 'bid', 'multiplier', 'bombs', 'redeals', 'must_bid', 'leading', 'last', 'log')
    out = {k: copy.deepcopy(s[k]) for k in keys if k in s}
    out['players'] = [{k: p[k] for k in ('count', 'bid', 'landlord', 'auto') if k in p}
                      for p in out.get('players', [])]
    out['log'] = [{k: e[k] for k in ('kind', 'seat', 'cards', 'pattern', 'value') if k in e}
                  for e in out.get('log', [])]
    return out


def _one_hot(index, size):
    a = np.zeros(size, dtype=np.float32)
    if index is not None and 0 <= index < size: a[index] = 1
    return a


def _seat_vector(seat, me):
    return _one_hot((seat-me)%3 if isinstance(seat, int) and 0 <= seat < 3 else 3, 4)


def _rank_vector(ranks):
    return np.bincount(list(ranks), minlength=15).astype(np.float32)/4


def _card_vector(cards):
    return _rank_vector(map(rank, cards))


def _pattern_vector(pattern):
    p = pattern or {}
    one = _one_hot(TYPE_ORDER.index(p['type']) if p.get('type') in TYPE_ORDER else None, 14)
    return np.concatenate((one, [p.get('rank', 0)/14, p.get('len', 0)/12])).astype(np.float32)


def _target_cards(s):
    if s['phase'] != 'playing' or s.get('leading'): return []
    target = s.get('last') or {}
    entry = next((e for e in reversed(current_log(s)) if e.get('kind') == 'play'), None)
    # Even when cards are supplied, require the full public log for this model.
    if (entry is None or entry.get('seat') != target.get('seat')
            or entry.get('pattern') != target.get('pattern')):
        raise ValueError('公开出牌历史与桌面不一致，请 /refresh 同步完整状态。')
    return entry['cards']


def encode_history(s):
    """All ordered strategic events, including past redeals; never truncate.

    Event values describe the public position AFTER the event, except the two
    explicit before-action flags (free lead / compulsory bidding).
    """
    me = s['seat']
    log = [e for e in s.get('log', []) if e.get('kind') in EVENT_KINDS[1:]]
    first = next((e['seat'] for e in log if e['kind'] == 'bid'), s.get('turn'))
    events = [{'kind': 'start', 'seat': first}]+log
    counts, pc = [17]*3, [0]*3
    land = None; base = bombs = redeals = nbids = passes = deal_step = 0
    multiplier = 1; leading = True
    rows = []
    for index, e in enumerate(events):
        kind, seat = e['kind'], e.get('seat')
        lead_before = leading
        forced_before = kind == 'bid' and nbids == 2 and base == 0 and redeals >= 3
        cards = e.get('cards') or []
        pattern = e.get('pattern') or {}
        if kind == 'redeal':
            redeals += 1; counts, pc = [17]*3, [0]*3
            land = None; base = bombs = nbids = passes = deal_step = 0
            multiplier = 1; leading = True
        elif kind == 'bid':
            nbids += 1; base = max(base, int(e['value']))
        elif kind == 'landlord':
            land, base = seat, int(e['value']); counts[land] = 20
            leading = True
        elif kind == 'play':
            counts[seat] -= len(cards); pc[seat] += 1
            passes = 0; leading = False
            if pattern.get('type') in ('bomb', 'rocket'):
                bombs += 1; multiplier += 1
        elif kind == 'pass':
            passes += 1
            if passes == 2: leading = True
        elif kind == 'finish' and land is not None:
            spring = seat == land and all(pc[i] == 0 for i in range(3) if i != land)
            anti = seat != land and pc[land] == 1
            multiplier += int(spring or anti)
        if kind not in ('start', 'redeal'): deal_step += 1
        spring = land is not None and all(pc[i] == 0 for i in range(3) if i != land)
        anti = land is not None and pc[land] <= 1
        order = [me, (me+1)%3, (me+2)%3]
        row = np.concatenate((
            _one_hot(EVENT_KINDS.index(kind), 7), _seat_vector(seat, me),
            _card_vector(cards), _pattern_vector(pattern),
            _one_hot(int(e['value']) if kind in ('bid', 'landlord') else None, 4),
            _seat_vector(land, me), np.array(counts)[order]/20, np.array(pc)[order]/20,
            [base/3, multiplier/16, bombs/14, redeals/3, nbids/3, passes/2,
             float(lead_before), float(forced_before), float(spring), float(anti),
             index/192, deal_step/192]))
        rows.append(row)
    history = np.asarray(rows, dtype=np.float32)
    if history.shape[1] != EVENT_DIM: raise RuntimeError('历史编码维度错误。')
    return history


def prepare(s, actions):
    """Return state[264], history[T,68], candidate actions[N,57].

    Seats are relative to the acting player: self, next, previous. Ranks use
    exact counts (divided by four), not suits. Histories retain every pass.
    """
    me = s['seat']
    land = s.get('landlord') if s['phase'] != 'bidding' else None
    order = [me, (me+1)%3, (me+2)%3]
    logs = current_log(s)
    played, last_plays, last_kind = [[] for _ in range(3)], [[] for _ in range(3)], [0]*3
    bids = [-1]*3; pc = [0]*3; passes = 0; nbids = 0; public = []
    for e in logs:
        kind, seat = e.get('kind'), e.get('seat')
        if kind == 'bid':
            bids[seat] = int(e['value']); nbids += 1; last_kind[seat] = 1
        elif kind == 'play':
            cards = e['cards']; played[seat] += cards; public += cards
            last_plays[seat] = cards; last_kind[seat] = 2; pc[seat] += 1; passes = 0
        elif kind == 'pass':
            last_kind[seat] = 3; passes += 1
    players = s['players']
    # Preserve real 0 bids. QOJ uses null after bidding; logs are authoritative.
    for i in range(3):
        if bids[i] < 0 and players[i].get('bid') is not None:
            bids[i] = int(players[i]['bid'])
    bottom = s.get('bottom') or []
    if land is not None and not any(e.get('kind') == 'landlord' for e in logs):
        raise ValueError('缺少公开叫地主历史，请 /refresh 同步完整状态。')
    if land is None: bottom = []  # Never expose unrevealed bottom cards.
    own_start = set(s['hand']) | set(played[me])
    if land == me: own_start -= set(bottom)
    unseen = set(range(54))-set(s['hand'])-set(public)
    bottom_left = set(bottom)-set(public)
    target_cards = _target_cards(s)
    target_pattern = to_beat(s)
    first = next((e['seat'] for e in logs if e.get('kind') == 'bid'), s.get('turn'))
    lastseat = (s.get('last') or {}).get('seat')
    spring = land is not None and all(pc[i] == 0 for i in range(3) if i != land)
    anti = land is not None and pc[land] <= 1
    # Some online states reset the displayed redeal counter after bidding.
    redeals = max(int(s.get('redeals') or 0), sum(e.get('kind') == 'redeal' for e in s.get('log', [])))
    history = encode_history(s)
    card_features = [_card_vector(cs) for cs in
                     (s['hand'], own_start, unseen, bottom, bottom_left,
                      *(played[i] for i in order), *(last_plays[i] for i in order), target_cards)]
    scalars = [float(s.get('bid') or 0)/3, float(s.get('multiplier') or 1)/16,
               float(s.get('bombs') or 0)/14, redeals/3, float(bool(s.get('leading'))),
               float(bool(s.get('must_bid'))), nbids/3, passes/2,
               (len(history)-1)/192,
               sum(e.get('kind') in EVENT_KINDS[1:] for e in logs)/192,
               float(spring), float(anti), float(land is not None and pc[land] >= 1),
               float(bool(bottom)), len(bottom_left)/3, 1.0]
    state = np.concatenate((
        *card_features, _one_hot(('bidding', 'playing', 'finished').index(s['phase']), 3),
        _seat_vector(s.get('turn'), me), _seat_vector(land, me),
        _seat_vector(first, me), _seat_vector(lastseat, me),
        [players[i]['count']/20 for i in order],
        *[_one_hot(bids[i]+1, 5) for i in order], [pc[i]/20 for i in order],
        *[_one_hot(last_kind[i], 4) for i in order], _pattern_vector(target_pattern), scalars
    )).astype(np.float32)
    if state.shape != (STATE_DIM,): raise RuntimeError('局面编码维度错误。')
    hand = _card_vector(s['hand'])
    rows = []
    for a in actions:
        cards = _rank_vector(a.ranks)
        rows.append(np.concatenate((
            cards, hand-cards, _one_hot(('bid', 'play', 'pass').index(a.kind), 3),
            _pattern_vector(a.reading), _one_hot(a.value if a.kind == 'bid' else None, 4),
            [(len(s['hand'])-len(a.ranks))/20, len(a.ranks)/20,
             float(a.kind == 'play' and len(a.ranks) == len(s['hand'])),
             float(bool(a.pattern) and a.pattern[0] in ('bomb', 'rocket'))])))
    candidates = np.asarray(rows, dtype=np.float32).reshape(-1, ACTION_DIM)
    return state, history, candidates


class Policy(nn.Module):
    """One trainable network shared by bidding and all three player roles.

    GRU states for EVERY public event remain available to query attention;
    no sliding window and no hidden state reused between independent games.
    """
    def __init__(self):
        super().__init__()
        self.state_net = nn.Sequential(nn.Linear(STATE_DIM, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.event_net = nn.Sequential(nn.Linear(EVENT_DIM, 96), nn.ReLU())
        self.history_net = nn.GRU(96, 128, batch_first=True)
        self.query = nn.Linear(256, 128)
        self.fusion = nn.Sequential(nn.Linear(512, 256), nn.ReLU())
        self.action_net = nn.Sequential(nn.Linear(ACTION_DIM, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.value_net = nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        nn.init.normal_(self.value_net[-1].weight, std=0.01)
        nn.init.zeros_(self.value_net[-1].bias)

    def encode(self, state, history, lengths):
        x = self.state_net(state)
        sequence, _ = self.history_net(self.event_net(history))
        scores = (sequence*self.query(x)[:, None, :]).sum(-1)/(128**0.5)
        mask = torch.arange(history.shape[1], device=history.device)[None, :] < lengths[:, None]
        attention = torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=1)
        pool = (sequence*attention[:, :, None]).sum(1)
        last = sequence[torch.arange(len(lengths), device=lengths.device), lengths-1]
        return self.fusion(torch.cat((x, pool, last), dim=-1))

    def score(self, encoded, actions):
        return self.value_net(torch.cat((encoded, self.action_net(actions)), dim=-1)).squeeze(-1)

    def forward(self, state, history, lengths, actions):
        return self.score(self.encode(state, history, lengths), actions)


# CPU model bundles and atomic persistence

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
    return {'schema': SCHEMA, 'architecture': dict(ARCHITECTURE),
            'weights': {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}, 'meta': dict(meta)}


def check_bundle(data):
    if data.get('schema') != SCHEMA or data.get('architecture') != ARCHITECTURE:
        raise ValueError('模型格式不兼容；final-v2 只接受新模型，不能加载 final-v1 权重。')


def load_bundle(path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    check_bundle(data)
    policy = Policy(); policy.load_state_dict(data['weights'], strict=True); policy.eval()
    return policy, data


class Agent:
    def __init__(self, path, threads=2):
        torch.set_num_threads(threads)
        self.path = Path(path)
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
        state, history, candidates = prepare(s, actions)
        encoded = self.policy.encode(torch.from_numpy(state[None]), torch.from_numpy(history[None]),
                                     torch.tensor([len(history)], dtype=torch.long))
        values = []
        for start in range(0, len(actions), 256):
            a = torch.from_numpy(candidates[start:start+256])
            values.append(self.policy.score(encoded.expand(len(a), -1), a).numpy())
        result = np.concatenate(values)
        return (result, (state, history, candidates)) if return_features else result

    def recommend(self, s, top=5):
        self.reload()
        actions = legal_actions(s)
        if not actions: raise ValueError('没有合法动作。')
        values = self.values(s, actions)
        order = np.argsort(-values, kind='stable')[:top]
        return {'action': actions[int(order[0])].to_dict(),
                'suggestions': [{'action': actions[int(i)].to_dict(), 'value': float(values[i])*SCORE_SCALE} for i in order],
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
                    rows.append('估值是预计本局积分，不是胜率；随机初始模型的估值没有实际参考意义。')
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
