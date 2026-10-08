#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 基于 EricWan377873/qoj-ddz 的 final-douzero 在线客户端整理。
# 原工程的第三方致谢及 GPL-3.0 / Apache-2.0 许可见原仓库 LICENSE.txt。
# DouZero: 718a5c920bf3361e34178a38f3b80458e176b351 (Apache-2.0)。
# AlphaDou: 13e740c08c3b653c2bef6ca345fc8fa6adc7d362 (GPL-3.0)。
"""QOJ 斗地主在线客户端（单文件版）。

保留单局、九局记分比赛、手动操作、规则提示、服务器托管和终端 UI。
不包含本地模型、训练、自博弈或本地对战引擎。

安装：python -m pip install curl_cffi beautifulsoup4 prompt_toolkit
启动：python qoj_cli.py
可选浏览器验证：python -m pip install playwright
                python -m playwright install chromium
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

ORIGIN = "https://qoj.ac"
LOBBY = "/games/doudizhu"
API = LOBBY + "/api"
COOKIE_NAME = "__Host-UOJSESSID"
RANKS = "3456789XJQKA2SD"  # X=10, S=小王, D=大王
TYPE_ORDER = ("rocket", "bomb", "single", "pair", "trio", "straight", "pairs",
              "plane", "trio1", "trio2", "four2", "four22", "plane1", "plane2")
TYPE_NAMES = dict(zip(TYPE_ORDER, ("王炸", "炸弹", "单张", "对子", "三张", "顺子",
                                   "连对", "飞机", "三带一", "三带一对", "四带二",
                                   "四带两对", "飞机带单", "飞机带对")))
INTERMISSION_SECONDS = 5.0


class ClientError(Exception):
    pass


class LoginError(ClientError):
    pass


class NetworkError(ClientError):
    """写请求可能已到达服务器，不得自动重试。"""


class ApiError(ClientError):
    def __init__(self, message, payload=None, status=0, retry_after=0):
        super().__init__(message)
        self.payload = payload or {}
        self.status = status
        self.retry_after = retry_after


def clean(value):
    """避免服务端文字携带终端控制符。"""
    return "".join(c if unicodedata.category(c) not in ("Cc", "Cf", "Cs") else " "
                   for c in str(value))


def parse_cookie(raw):
    raw = raw.strip()
    if COOKIE_NAME + "=" in raw:
        raw = raw.split(COOKIE_NAME + "=", 1)[1].split(";", 1)[0].strip()
    if not raw or not re.fullmatch(r"[A-Za-z0-9,._~%+/-]+", raw):
        raise ClientError("请输入 __Host-UOJSESSID 的值或包含该字段的 Cookie 文本。")
    return raw


def rank(card):
    return card // 4 if card < 52 else card - 39


def cards_text(cards, grouped=False, desc=False):
    counts = Counter(rank(c) for c in cards or [])
    chunks = [RANKS[r] * counts[r] for r in sorted(counts, reverse=desc)]
    return (" " if grouped else "").join(chunks) or "∅"


def parse_ranks(text):
    text = unicodedata.normalize("NFKC", text).strip().upper()
    text = text.replace("小王", "S").replace("大王", "D").replace("10", "X").replace("T", "X")
    text = re.sub(r"[\s,，]+", "", text)
    if not text or any(c not in RANKS for c in text):
        return None
    return [RANKS.index(c) for c in text]


def select_cards(hand, wanted):
    groups = {r: [] for r in range(15)}
    for card in sorted(hand or []):
        groups[rank(card)].append(card)
    chosen = []
    for r, n in Counter(wanted).items():
        if len(groups[r]) < n:
            raise ClientError(f"手中没有 {n} 张 {RANKS[r]}。使用 /say 可强制聊天。")
        chosen.extend(groups[r][:n])
    return sorted(chosen)


def classify(cards):
    """使用 QOJ 的牌型语义，歧义牌型由使用者显式选择。"""
    n = len(cards)
    if not n or len(set(cards)) != n or any(not 0 <= x < 54 for x in cards):
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
    for kind, size, minimum in (("straight", 1, 5), ("pairs", 2, 3), ("plane", 3, 2)):
        rs = [r for r, v in enumerate(c) if v]
        if (len(rs) >= minimum and rs[-1] <= 11 and
                rs[-1] - rs[0] == len(rs) - 1 and all(c[r] == size for r in rs)):
            add(kind, rs[-1], len(rs))
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


def describe(pattern):
    if not pattern:
        return ""
    return TYPE_NAMES.get(pattern["type"], pattern["type"]) + (
        "" if pattern["type"] == "rocket" else " " + RANKS[pattern["rank"]])


def pattern_key(pattern):
    return f"{pattern['type']}:{pattern['rank']}:{pattern['len']}"


def hints(hand, last):
    """沿用原项目规则提示，不使用神经网络，不自动出牌。"""
    g = [[] for _ in range(15)]
    for card in sorted(hand):
        g[rank(card)].append(card)
    cand = []

    def broken(r, count):
        return int(bool(g[13] and g[14])) if r >= 13 else int(len(g[r]) > count)

    def kickers(exclude, count, unit):
        rs = [r for r in range(15) if r not in exclude and len(g[r]) >= unit
              and (unit != 2 or r < 13)]
        rs.sort(key=lambda r: (broken(r, unit), r))
        if unit == 2:
            return [c for r in rs[:count] for c in g[r][:2]] if len(rs) >= count else None
        cards = [g[r][layer] for layer in range(3) for r in rs if len(g[r]) > layer]
        return cards[:count] if len(cards) >= count else None

    if not last:
        for r, group in enumerate(g):
            if group and not (r >= 13 and g[13] and g[14]):
                cand.append(((int(len(group) == 4), r), group))
    else:
        kind, top, length = last["type"], last["rank"], last["len"]
        size = {"single": 1, "pair": 2, "trio": 3, "trio1": 3, "trio2": 3}.get(kind)
        if size:
            for r in range(top + 1, 15):
                if len(g[r]) < size:
                    continue
                body = g[r][:size]
                if kind in ("trio1", "trio2"):
                    extra = kickers({r}, 1, 1 if kind == "trio1" else 2)
                    if extra is None:
                        continue
                    body += extra
                cand.append(((0, broken(r, size), r), body))
        size = {"straight": 1, "pairs": 2, "plane": 3, "plane1": 3, "plane2": 3}.get(kind)
        if size:
            for end in range(top + 1, 12):
                start = end - length + 1
                if start < 0 or any(len(g[r]) < size for r in range(start, end + 1)):
                    continue
                body = [card for r in range(start, end + 1) for card in g[r][:size]]
                if kind in ("plane1", "plane2"):
                    extra = kickers(set(range(start, end + 1)), length,
                                    1 if kind == "plane1" else 2)
                    if extra is None:
                        continue
                    body += extra
                cand.append(((0, int(any(broken(r, size) for r in range(start, end + 1))), end), body))
        if kind in ("four2", "four22"):
            for r in range(top + 1, 13):
                if len(g[r]) == 4:
                    extra = kickers({r}, 2, 1 if kind == "four2" else 2)
                    if extra is not None:
                        cand.append(((0, 0, r), g[r] + extra))
        if kind != "rocket":
            for r in range(13):
                if len(g[r]) == 4 and (kind != "bomb" or r > top):
                    cand.append(((1, 0, r), g[r]))
    if g[13] and g[14] and (not last or last["type"] != "rocket"):
        cand.append(((2, 0, 14), [52, 53]))
    result, seen = [], set()
    for _, cards in sorted(cand, key=lambda pair: (len(pair[0]), *pair[0])):
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
            raise LoginError("Cookie 无效或已过期。")
        raise ClientError("没有找到 #ddz-lobby，QOJ 页面结构可能已变。")
    stats = {}
    for node in soup.select(".card-body .text-muted.small"):
        label = node.get_text(strip=True)
        value = node.parent.select_one(".h4")
        if value and label in ("总积分", "对局", "胜率", "地主胜率"):
            stats[label] = value.get_text(strip=True)
    hero = soup.select_one(".ddz-lobby-hero")
    rating = hero.select_one("p strong") if hero else None
    history = soup.find("a", href=re.compile(r"/games/doudizhu/history\?user="))
    username = parse_qs(urlparse(history["href"]).query).get("user", [""])[0] if history else ""
    token = root.get("data-token", "")
    if not token:
        raise LoginError("大厅没有 CSRF token，请检查 Cookie。")
    return LobbyInfo(token, username, rating.get_text(strip=True) if rating else "—",
                     stats, int(root["data-game"]) if root.get("data-game") else None)


def parse_game(html):
    soup = soup_of(html)
    root, initial = soup.select_one("#ddz"), soup.select_one("#ddz-initial")
    if root is None or initial is None:
        if soup.select_one('input[type="password"]'):
            raise LoginError("登录已过期，请重新运行并输入 Cookie。")
        raise ClientError("没有找到 #ddz / #ddz-initial，页面结构可能已变。")
    data = json.loads(initial.get_text())
    if root.get("data-logged-in") != "1" or data["state"].get("seat") is None:
        raise LoginError("该 Cookie 不是这局的参赛者。")
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
    return reply.headers.get("cf-mitigated") == "challenge" or (is_html and any(
        x in head for x in ("<title>just a moment", "window._cf_chl_opt",
                         "/cdn-cgi/challenge-platform/", "cf-chl-widget")))


class Transport:
    """与 QOJ 的 HTTP/Cloudflare 浏览器回退通信，写操作不自动重试。"""

    def __init__(self, cookie, mode="auto", channel="auto", timeout=12, notify=print):
        from curl_cffi import requests

        self.cookie = cookie
        self.session = requests.Session(impersonate="chrome", timeout=timeout,
                                        headers={"Accept-Language": "zh-CN,zh;q=0.9"})
        self.session.cookies.jar.set_cookie(Cookie(
            0, COOKIE_NAME, cookie, None, False, "qoj.ac", False, False,
            "/", True, True, None, True, None, None, {"HttpOnly": None}, False))
        self.mode, self.channel, self.timeout = mode, channel, timeout
        self.notify = notify
        self.browser = self.context = self.page = self.pw = None
        self.browser_attempted = False
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qoj-http")
        self.lock = asyncio.Lock()

    @property
    def label(self):
        return "浏览器连接" if self.page else "HTTP 连接"

    async def _direct(self, method, path, data=None):
        def perform():
            url = ORIGIN + path
            for _ in range(4):
                response = self.session.request(
                    method, url, data=data, allow_redirects=False,
                    headers={"Referer": ORIGIN + LOBBY, "Origin": ORIGIN})
                if response.status_code in (301, 302, 303, 307, 308):
                    target = urljoin(url, response.headers.get("location", ""))
                    parsed = urlparse(target)
                    if parsed.scheme != "https" or parsed.netloc != "qoj.ac":
                        raise ClientError("QOJ 返回非本站重定向，已停止请求。")
                    if method != "GET":
                        raise LoginError("写接口返回重定向，请重新确认登录。")
                    url = target
                    continue
                return Reply(response.status_code, response.text,
                             dict(response.headers), str(response.url))
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
            raise ClientError("浏览器验证需要 pip install playwright，随后运行 python -m playwright install chromium。") from None
        self.notify("请在弹出的浏览器中完成验证，并保持窗口打开。")
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
            raise ClientError("浏览器启动失败；请安装 Chromium 和系统图形依赖。")
        self.context = await self.browser.new_context(locale="zh-CN")
        await self.context.add_cookies([{
            "name": COOKIE_NAME, "value": self.cookie, "url": ORIGIN + "/", "secure": True
        }])
        # 避免网页 JavaScript 和本 CLI 对同一局重复发请求。
        await self.context.route(
            re.compile(r"https://qoj\.ac/js/games/doudizhu(?:-lobby|-rules)?\.js(?:\?.*)?$"),
            lambda route: route.abort())
        self.page = await self.context.new_page()
        await self.verify_browser()

    async def verify_browser(self):
        try:
            await self.page.goto(ORIGIN + LOBBY + "?locale=zh-cn",
                                 wait_until="domcontentloaded", timeout=30000)
        except Exception:
            pass
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            try:
                if self.page.is_closed():
                    break
                if await self.page.locator("#ddz-lobby").count():
                    self.notify("浏览器验证通过。")
                    return
                if await self.page.locator('input[type="password"]').count():
                    raise LoginError("Cookie 无效或已过期。")
            except LoginError:
                raise
            except Exception:
                pass
            await asyncio.sleep(0.6)
        raise ClientError("浏览器验证未完成；可用 /browser 重试。")

    async def _browser_request(self, method, path, data=None):
        try:
            response = await self.page.evaluate("""async ({method,path,data,timeout}) => {
                const controller = new AbortController();
                const timer = setTimeout(() => controller.abort(), timeout);
                try {
                    const opts = {method, credentials:'same-origin', cache:'no-store',
                                  redirect:'error', signal:controller.signal};
                    if (method === 'POST') opts.body = new URLSearchParams(data);
                    const r = await fetch(path, opts);
                    return {status:r.status, text:await r.text(),
                            headers:Object.fromEntries(r.headers), url:r.url};
                } finally { clearTimeout(timer); }
            }""", {"method": method, "path": path, "data": data,
                   "timeout": self.timeout * 1000})
            return Reply(**response)
        except Exception:
            raise NetworkError("浏览器请求失败；请保持验证窗口打开。") from None

    async def request(self, method, path, data=None):
        if not path.startswith(LOBBY) or urlparse(path).netloc:
            raise ClientError("拒绝访问 QOJ 斗地主以外的路径。")
        async with self.lock:
            if self.mode == "browser" and not self.page:
                if self.browser_attempted:
                    raise ClientError("浏览器未就绪，请手动输入 /browser。")
                await self.enable_browser()
            request = self._browser_request if self.page else self._direct
            response = await request(method, path, data)
            if is_challenge(response):
                if self.mode == "http":
                    raise ClientError("Cloudflare 要求验证，请使用 /browser 或 --transport browser。")
                if self.browser_attempted:
                    raise ClientError("Cloudflare 仍要求验证，请输入 /browser。")
                await self.enable_browser()
                if method == "GET" or (data or {}).get("action") in ("state", "lobby"):
                    response = await self._browser_request(method, path, data)
                else:
                    raise NetworkError("已切换浏览器；先前写操作的结果不确定，未自动重发。")
            return response

    async def close(self):
        for obj, action in ((self.context, "close"), (self.browser, "close"), (self.pw, "stop")):
            if obj:
                with contextlib.suppress(Exception):
                    await getattr(obj, action)()
        with contextlib.suppress(Exception):
            await asyncio.get_running_loop().run_in_executor(self.executor, self.session.close)
        self.executor.shutdown(wait=False)


def match_of(state):
    return (state or {}).get("match") or {}


def between_rounds(state):
    return bool(state and state.get("phase") == "finished" and match_of(state)
                and not match_of(state).get("finished"))


def own_name(state):
    if not state:
        return ""
    me = state.get("seat")
    players = state.get("players") or []
    return players[me].get("username", "") if isinstance(me, int) and 0 <= me < len(players) else ""


class QojClient:
    """统一处理单局/记分赛，无运行时 monkey patch。"""

    def __init__(self, transport):
        self.transport = transport
        self.lobby = LobbyInfo()
        self.lobby_status = {}
        self.queue_mode = "single"
        self.state = None
        self.token = ""
        self.chat = {}
        self.chat_after = 0
        self.commitments = {}
        self.queue_may_be_active = False

    async def get_html(self, path):
        response = await self.transport.request("GET", path)
        if not 200 <= response.status < 300 or is_challenge(response):
            raise ApiError(f"页面请求失败（HTTP {response.status}）；可用 /browser 验证。",
                           status=response.status)
        return response.text

    async def load_lobby(self):
        info = parse_lobby(await self.get_html(LOBBY + "?locale=zh-cn"))
        self.lobby, self.token = info, info.token
        self.lobby_status = {"game": info.game, "queued": None}
        self.state = None

    async def load_game(self, game_id, *, expected_match=None, after_round=None):
        token, data = parse_game(await self.get_html(
            f"{LOBBY}/game/{int(game_id)}?locale=zh-cn"))
        state = data["state"]
        if expected_match is not None:
            m = match_of(state)
            if (state.get("id") != int(game_id) or m.get("id") != expected_match or
                    int(m.get("round") or 0) <= int(after_round or 0)):
                raise ClientError("下一局尚未准备好，继续显示旧分表。")
            if own_name(state) != own_name(self.state):
                raise ClientError("下一局参赛身份不一致，保留旧分表。")
        self.token = token
        self.state = None
        self.chat, self.chat_after, self.commitments = {}, 0, {}
        self.accept(data)

    def accept(self, data):
        if isinstance(data.get("lobby"), dict):
            self.lobby_status = data["lobby"]
            self.queue_may_be_active = bool(self.lobby_status.get("queued"))
        state = data.get("state")
        if isinstance(state, dict):
            if state.get("unchanged"):
                # 服务器可能只更新比赛元数据，不能覆盖完整牌局状态。
                m = state.get("match")
                if (self.state and isinstance(m, dict) and
                        m.get("id") == match_of(self.state).get("id")):
                    self.state = {**self.state, "match": m}
            elif (self.state is None or state.get("id") != self.state.get("id") or
                  state.get("version", 0) >= self.state.get("version", 0)):
                self.state = state
                for i, digest in enumerate((state.get("fairness") or {}).get("commitments") or []):
                    self.commitments.setdefault(i, digest)
        if self.state:
            self.queue_mode = "match" if match_of(self.state) else "single"
        elif self.lobby_status.get("queued") in ("single", "match"):
            self.queue_mode = self.lobby_status["queued"]
        for message in data.get("chat") or []:
            if isinstance(message.get("id"), int):
                self.chat[message["id"]] = message
                self.chat_after = max(self.chat_after, message["id"])

    async def call(self, action, **extra):
        body = {"_token": self.token, "action": action}
        if action in ("queue", "cancel", "lobby"):
            mode = extra.pop("mode", self.queue_mode)
            if mode not in ("single", "match"):
                raise ClientError("匹配模式必须是 single 或 match。")
            self.queue_mode = mode
            body["mode"] = mode
        else:
            if not self.state:
                raise ClientError("请先进入对局。")
            body.update(game=str(self.state["id"]), version=str(self.state["version"]),
                        chat_after=str(self.chat_after))
        body.update({k: str(v) for k, v in extra.items()})
        if action == "queue":
            self.queue_may_be_active = True
        response = await self.transport.request("POST", API, body)
        try:
            data = json.loads(response.text)
            if not isinstance(data, dict):
                raise ValueError
        except (ValueError, TypeError):
            if "/login" in response.url or 'type="password"' in response.text:
                raise LoginError("登录已过期，请重新输入 Cookie。") from None
            raise ApiError(f"接口未返回 JSON（HTTP {response.status}）。",
                           status=response.status) from None
        self.accept(data)  # 错误响应也可能带有权威新状态
        if not 200 <= response.status < 300 or data.get("error"):
            retry = response.headers.get("retry-after", "0")
            raise ApiError(clean(data.get("error") or f"HTTP {response.status}"),
                           data, response.status,
                           min(60, int(retry)) if str(retry).isdigit() else 0)
        return data

    async def refresh_token(self):
        if self.state:
            self.token, _ = parse_game(await self.get_html(
                f"{LOBBY}/game/{self.state['id']}?locale=zh-cn"))
        else:
            self.lobby = parse_lobby(await self.get_html(LOBBY + "?locale=zh-cn"))
            self.token = self.lobby.token


def name_of(state, seat):
    players = (state or {}).get("players") or []
    return clean(players[seat].get("username", "")) if isinstance(seat, int) and 0 <= seat < len(players) else "—"


def seat_order(state):
    me = state.get("seat")
    return [(me + 1) % 3, (me + 2) % 3, me] if isinstance(me, int) and 0 <= me < 3 else [0, 1, 2]


def to_beat(state):
    if state and state.get("phase") == "playing" and not state.get("leading"):
        return (state.get("last") or {}).get("pattern")
    return None


def compact_log(state):
    rows = []
    for event in state.get("log") or []:
        kind = event.get("kind")
        if kind == "redeal":
            rows.append("重新发牌")
            continue
        who = name_of(state, event.get("seat"))
        if kind == "play":
            msg = cards_text(event.get("cards"))
        elif kind == "pass":
            msg = "jump"
        elif kind == "bid":
            msg = f"叫{event['value']}" if event.get("value") else "不叫"
        elif kind == "landlord":
            msg = f"地主{event.get('value')} 底{cards_text(event.get('cards'))}"
        elif kind == "auto_on":
            msg = "超时托管" if event.get("timeout") else "托管"
        elif kind == "auto_off":
            msg = "取消托管"
        elif kind == "finish":
            msg = "出完"
        else:
            msg = clean(kind or "事件")
        rows.append(f"{who}:{msg}" + ("*" if event.get("auto") else ""))
    return rows


def pack_logs(entries, columns=100):
    from prompt_toolkit.utils import get_cwidth
    rows, current = [], []
    for entry in entries:
        if current and (len(current) >= 3 or
                        get_cwidth("  |  ".join(current + [entry])) > max(20, columns - 2)):
            rows.append("  |  ".join(current))
            current = []
        current.append(entry)
    if current:
        rows.append("  |  ".join(current))
    return rows


def counter_text(state):
    left = [4] * 13 + [1, 1]
    for event in state.get("log") or []:
        if event.get("kind") == "play":
            for card in event.get("cards") or []:
                left[rank(card)] -= 1
    for card in state.get("hand") or []:
        left[rank(card)] -= 1
    return " ".join(f"{RANKS[r]}:{max(0, left[r])}" for r in range(14, -1, -1))


def fairness_lines(client, detail=False):
    info = (client.state or {}).get("fairness") or {}
    commits, deals = info.get("commitments") or [], info.get("deals") or []
    if not commits:
        return ["服务器未提供发牌承诺。"]
    if not deals and not detail:
        return [f"发牌公平性：已记录 {len(commits)} 份 SHA-256 承诺（/fair 详情）"]
    rows = []
    for i, commitment in enumerate(commits):
        note = "等待终局公开"
        deal = deals[i] if i < len(deals) else None
        if deal:
            try:
                deck = deal["deck"]
                digest = hashlib.sha256((",".join(map(str, deck)) + "|" + deal["salt"]).encode()).hexdigest()
                valid = sorted(deck) == list(range(54))
                valid &= digest == commitment == client.commitments.get(i, commitment) == deal.get("commitment")
                note = "校验通过" if valid else "校验不一致！"
            except (KeyError, TypeError):
                note = "公开格式异常"
        rows.append(f"发牌 {i+1}：{commitment if detail else commitment[:12]+'…'} {note}")
        if detail and deal:
            rows += ["deck=" + ",".join(map(str, deal.get("deck") or [])),
                     "salt=" + clean(deal.get("salt", ""))]
    return rows


def match_player_order(state):
    names = match_of(state).get("players") or []
    players = (state or {}).get("players") or []
    indices = []
    if state and isinstance(state.get("seat"), int) and len(players) == 3:
        for seat in seat_order(state):
            name = players[seat].get("username")
            if name in names and names.index(name) not in indices:
                indices.append(names.index(name))
    return indices + [i for i in range(len(names)) if i not in indices]


def signed(v):
    return f"{v:+}" if isinstance(v, (int, float)) else "—"


def at(values, i, default=None):
    return values[i] if isinstance(values, (list, tuple)) and isinstance(i, int) and 0 <= i < len(values) else default


def text_width(text):
    try:
        from prompt_toolkit.utils import get_cwidth
        return get_cwidth(text)
    except ImportError:
        return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def fitted(value, size, right=False):
    value = clean(value)
    if text_width(value) > size:
        out = ""
        for c in value:
            if text_width(out + c + "…") > size:
                break
            out += c
        value = out + "…"
    spaces = " " * max(0, size - text_width(value))
    return spaces + value if right else value + spaces


def match_summary(state):
    match = match_of(state)
    names = match.get("players") or []
    totals = match.get("totals") or []
    return "累计分：" + "  ".join(
        f"{clean(names[i])}{' [我]' if names[i] == own_name(state) else ''} {signed(at(totals, i))}"
        for i in match_player_order(state))


def scoreboard_lines(state, columns=100):
    m = match_of(state)
    if not m:
        return ["当前没有记分比赛。"]
    names = m.get("players") or []
    rounds = max(1, min(99, int(m.get("rounds") or 9)))
    finished = bool(m.get("finished"))
    rows = [f"记分比赛 #{m.get('id', '—')} · " +
            ("比赛结束" if finished else f"第 {m.get('round', '—')}/{rounds} 局"), ""]
    deltas, totals = m.get("deltas") or [], m.get("totals") or []
    labels = [clean(name) + (" [我]" if name == own_name(state) else "") for name in names]
    name_size = max(10, min(26, max((text_width(x) for x in labels), default=10)))
    cell_size = max(3, max((text_width(signed(at(delta, i))) for delta in deltas
                            for i in range(len(names))), default=3))
    total_size = max(4, max((text_width(signed(x)) for x in totals), default=4))
    per_block = max(1, min(rounds, (max(40, columns) - name_size - total_size - 4) // (cell_size + 1)))
    for start in range(0, rounds, per_block):
        end = min(rounds, start + per_block)
        head = fitted("玩家 / 局", name_size) + "  "
        head += " ".join(fitted(str(r+1), cell_size, True) for r in range(start, end))
        rows.append(head + "  " + fitted("总分", total_size, True))
        for i in match_player_order(state):
            cols = [signed(at(at(deltas, r, []), i)) for r in range(start, end)]
            line = fitted(labels[i], name_size) + "  "
            line += " ".join(fitted(x, cell_size, True) for x in cols)
            rows.append(line + "  " + fitted(signed(at(totals, i)), total_size, True))
        rows.append("")
    if finished:
        rows.append("最终名次 / Rating")
        for i in match_player_order(state):
            place = at(m.get("places"), i)
            before, after = at(m.get("rating_before"), i), at(m.get("rating_after"), i)
            rating = (f"Rating {before} → {after} ({signed(after-before)})"
                      if isinstance(before, (int, float)) and isinstance(after, (int, float))
                      else f"Rating {after}" if after is not None else "Rating 待同步")
            rows.append(f"#{place if place is not None else '—'} {labels[i]} "
                        f"总分 {signed(at(totals, i))} {rating}")
    else:
        rows.append("最终名次和 Rating 将由服务器在整场比赛结束后公布。")
    result = (state or {}).get("result")
    if result:
        rows += ["", "本局：" + ("地主胜" if result.get("landlord_won") else "农民胜") +
                 f" · 底分 {result.get('base', '—')} × {result.get('multiplier', '—')}"]
    return rows


HELP = """QOJ 斗地主在线 CLI（无本地推理功能）
大厅：输入 1 单局匹配；2 同三人连续 9 局的记分比赛。
牌面：789XJQKA；X/T/10 都是 10；S/小王、D/大王 是两张王。
叫分阶段：0 或 - 不叫；1/2/3 叫分。出牌阶段 2、3 是牌。
直接输入牌面出牌，- 为过；非牌面普通文字发送聊天（最多 60 UTF-16 字符）。
/say 文本 强制聊天；/play 牌面 强制出牌。
/choose 编号 或 牌面#编号：选择有歧义的合法牌型。
/hint 查看纯规则提示（不会出牌）；/auto on|off：QOJ 服务器托管。
/mute 屏蔽或恢复他人聊天；/sort 切换手牌排序。
/log、/chat 查看全部日志及聊天；/fair 查看发牌校验。
/initial：终局后查看服务器公开的三人开局手牌。
/score 或 /scores：查看记分比赛分表（仅比赛中）。
/cancel 取消匹配；/again 终局后再次匹配；/home 返回大厅。
/refresh 刷新；/browser 浏览器验证；/help 帮助；/quit 退出。
Esc 或 /close 关闭详情；F2/F3 向前翻日志/聊天；PgUp/PgDn 滚动详情。
写请求超时不会重发；请用 /refresh 检查服务器状态。"""


@dataclass(frozen=True)
class Submitted:
    line: str
    game_id: int | None
    during_break: bool = False


class TerminalUI:
    """一个可刷新全屏输入框；比赛与单局共享同一套命令处理。"""

    def __init__(self, client, demo=False, input=None, output=None):
        from prompt_toolkit.application import Application
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.document import Document
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import HSplit, VSplit, Layout, Window, DynamicContainer
        from prompt_toolkit.layout.containers import ConditionalContainer
        from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
        from prompt_toolkit.styles import Style
        from prompt_toolkit.widgets import TextArea

        self.client, self.demo = client, demo
        self.notice = "离线界面演示；不会发送网络请求。" if demo else "正在连接…"
        self.connected = demo
        self.busy = self.quitting = self.muted = self.desc = False
        self.detail = ""
        self.pending = None
        self.scroll = {"log": 0, "chat": 0}
        self.op_lock = asyncio.Lock()
        self.commands = asyncio.Queue(maxsize=1)
        self.wakeup = asyncio.Event()
        self.break_game = None
        self.break_deadline = 0.0
        self.Document = Document
        self.details = TextArea(read_only=True, scrollbar=True, wrap_lines=True)

        kb = KeyBindings()

        @kb.add("c-c")
        @kb.add("c-d")
        def quit_key(event):
            self.quitting = True
            event.app.exit()

        @kb.add("escape", eager=True, is_global=True)
        def escape(event):
            self.detail = ""
            self.scroll = {"log": 0, "chat": 0}
            event.app.layout.focus(self.input_control)
            event.app.invalidate()

        @kb.add("f2")
        def scroll_log(event):
            self.scroll["log"] += 3
            event.app.invalidate()

        @kb.add("f3")
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
            line = buffer.text.strip()
            if not line:
                return False
            if self.busy or not self.commands.empty():
                self.set_notice("上一条输入仍在处理，当前输入已保留。")
                return True
            state = self.client.state
            self.commands.put_nowait(Submitted(line, (state or {}).get("id"), between_rounds(state)))
            return False

        self.buffer = Buffer(accept_handler=submit, multiline=False)
        self.input_control = BufferControl(self.buffer)
        self.log_window = Window(FormattedTextControl(self.render_log),
                                 height=lambda: 2 if self.small_screen() else 4, wrap_lines=True)
        self.chat_window = Window(FormattedTextControl(self.render_chat),
                                  height=lambda: 1 if self.small_screen() else 3, wrap_lines=True)
        self.board_window = Window(FormattedTextControl(self.render_board), wrap_lines=True)

        header = lambda: Window(FormattedTextControl(self.render_header), height=1, style="class:header")
        prompt = lambda: Window(FormattedTextControl(self.render_prompt), height=1, wrap_lines=True)
        notice = lambda: Window(FormattedTextControl(lambda: clean(self.notice)),
                                height=lambda: 1 if self.small_screen() else 2,
                                wrap_lines=True, style="class:notice")
        input_row = lambda: VSplit([
            Window(FormattedTextControl("> "), width=2, height=1),
            Window(self.input_control, height=1, wrap_lines=False, always_hide_cursor=False)
        ], style="class:input")
        section = lambda label: Window(FormattedTextControl(label), height=1, style="class:section")
        home = Condition(lambda: self.client.state is None and not self.detail)
        playing = Condition(lambda: self.client.state is not None and not self.detail)
        detail = Condition(lambda: bool(self.detail))
        normal = HSplit([
            header(),
            ConditionalContainer(Window(FormattedTextControl(self.render_home), wrap_lines=True), home),
            ConditionalContainer(HSplit([
                section(" 出牌日志（F2 向前 / Esc 最新）"), self.log_window,
                section(" 聊天（F3 向前 / Esc 最新）"), self.chat_window,
                section(" 牌桌"), self.board_window,
            ]), playing),
            ConditionalContainer(self.details, detail),
            ConditionalContainer(Window(FormattedTextControl(self.render_hand),
                                        height=lambda: 2 if self.small_screen() else 3,
                                        wrap_lines=True, style="class:hand"),
                                 Condition(lambda: self.client.state is not None)),
            notice(), prompt(), input_row(),
        ])
        score = HSplit([
            header(), section(" 九局比赛分表"),
            Window(FormattedTextControl(self.render_score), wrap_lines=True),
            notice(), prompt(), input_row(),
        ])

        def layout_root():
            state = self.client.state
            if (state and state.get("phase") == "finished" and match_of(state)
                    and not self.detail):
                return score
            return normal

        self.app = Application(
            layout=Layout(DynamicContainer(layout_root), self.input_control),
            key_bindings=kb, full_screen=True, mouse_support=False, input=input, output=output,
            style=Style.from_dict({"header": "bg:#17324d #ffffff bold",
                                   "section": "#66c2ff bold", "hand": "#ffdc75 bold",
                                   "notice": "#a7dba5", "input": "bg:#203040 #ffffff"}))

    def set_notice(self, message):
        self.notice = clean(message)
        self.app.invalidate()

    def small_screen(self):
        return self.app.output.get_size().rows < 30

    def render_header(self):
        s = self.client.state
        if s:
            m = match_of(s)
            context = f"比赛 #{m['id']} {m.get('round','—')}/{m.get('rounds',9)} 局" if m else "单局匹配"
            place = f"对局 #{s['id']}"
        else:
            context, place = "单局 / 九局比赛", "大厅"
        return f" QOJ 斗地主 | {context} | {place} | {'演示' if self.demo else self.client.transport.label} | /help"

    def render_home(self):
        info, status = self.client.lobby, self.client.lobby_status
        rows = ["", f"  玩家：{clean(info.username) or '—'}    Rating：{clean(info.rating)}", ""]
        rows += ["  " + "    ".join(f"{k}：{clean(info.stats.get(k, '—'))}"
                                  for k in ("总积分", "对局", "胜率", "地主胜率")), ""]
        if status.get("queued"):
            mode = status["queued"]
            size = (status.get("queue_size") or {}).get(mode, "—")
            rows += [f"  {'单局' if mode == 'single' else '九局比赛'}匹配中，队列 {size} 人",
                     "  /cancel 取消匹配"]
        elif status.get("game"):
            rows += [f"  正在进行的对局 #{status['game']} · 输入 1 返回对局"]
        else:
            rows += ["  1  单局匹配", "  2  记分比赛（9 局）"]
        rows += ["", "  /refresh 同步    /browser 浏览器验证    /quit 退出"]
        return "\n".join(rows)

    def _tail(self, entries, key, height):
        end = max(1, len(entries) - self.scroll[key])
        return "\n".join(entries[max(0, end-height):end]) if entries else "（暂无）"

    def render_log(self):
        s = self.client.state
        return self._tail(pack_logs(compact_log(s), self.app.output.get_size().columns),
                          "log", 2 if self.small_screen() else 4) if s else ""

    def render_chat(self):
        s = self.client.state
        if not s:
            return ""
        rows = [f"{clean(m.get('username',''))}: {clean(m.get('text',''))}"
                for _, m in sorted(self.client.chat.items())
                if not self.muted or m.get("seat") == s.get("seat")]
        return self._tail(rows, "chat", 1 if self.small_screen() else 3)

    def player_row(self, seat, mine=False):
        s = self.client.state
        player = s["players"][seat]
        marker = "> " if s.get("turn") == seat and s.get("phase") != "finished" else "  "
        tags = (" (地主)" if s.get("landlord") == seat else "") + (" [我]" if mine else "")
        tags += " [托管]" if player.get("auto") else ""
        bid = player.get("bid")
        bid_text = ((" · 不叫" if bid == 0 else f" · 叫{bid}分")
                    if bid is not None and s.get("phase") == "bidding" else "")
        table = at(s.get("table"), seat)
        played = ("jump" if table == "pass" else
                  cards_text(table.get("cards")) + " " + describe(table.get("pattern"))
                  if isinstance(table, dict) else "—")
        line = f"{marker}{name_of(s, seat)}{tags} · 剩 {player.get('count','—')} 张{bid_text} · 桌面 {played}"
        if not mine:
            hands = s.get("hands")
            line += "\n    手牌：" + (cards_text(hands[seat], True, self.desc)
                                     if hands is not None else "未公开")
        return line

    def render_board(self):
        s = self.client.state
        if not s:
            return ""
        phase = {"bidding": "叫分中", "playing": "出牌中", "finished": "已结束"}.get(s["phase"], s["phase"])
        rows = []
        if match_of(s):
            rows.append(match_summary(s) + "（/score 查看分表）")
        rows += [f"{phase} · 轮到 {name_of(s,s.get('turn'))} · 底分 {s.get('bid',0)} "
                 f"· ×{s.get('multiplier',1)} · 炸弹 {s.get('bombs',0)} · 流局 {s.get('redeals',0)}",
                 "底牌：" + (cards_text(s["bottom"]) if s.get("bottom") is not None else "未公开") +
                 (" · 自由出牌" if s.get("leading") else ""),
                 "未出现（不含我的手牌）：" + counter_text(s)]
        for seat in seat_order(s)[:-1]:
            rows.append(self.player_row(seat))
        result = s.get("result")
        if result:
            tags = (" · 春天" if result.get("spring") else "") + (" · 反春" if result.get("anti_spring") else "")
            scores = "  ".join(f"{name_of(s,i)} {delta:+}"
                               for i, delta in enumerate(result.get("deltas") or []))
            rows.append(f"结算：{'地主胜' if result.get('landlord_won') else '农民胜'}{tags} "
                        f"· {result.get('base','—')}×{result.get('multiplier','—')} · {scores}")
        rows += fairness_lines(self.client)
        return "\n".join(rows)

    def render_hand(self):
        s = self.client.state
        if not s:
            return ""
        return self.player_row(s["seat"], True) + "\n我的手牌：" + cards_text(s.get("hand"), True, self.desc)

    def render_score(self):
        s = self.client.state
        rows = scoreboard_lines(s, self.app.output.get_size().columns)
        rows += ["", "局间等待：准备好后自动进入下一局。" if between_rounds(s) else
                 "比赛结束：/again 再来一场；/home 回大厅。"]
        return "\n".join(rows)

    def render_prompt(self):
        s = self.client.state
        if not s:
            return "输入 1 单局；2 九局比赛；/help 帮助。"
        if between_rounds(s):
            return "正在等待下一局；/refresh 同步；/quit 退出。"
        if s.get("phase") == "finished":
            return "本场已结束：/again 再匹配；/home 大厅；可继续聊天。"
        if s.get("phase") == "bidding":
            return ("必须叫分：" if s.get("must_bid") else "叫分：0/- 不叫；") + "1/2/3 叫分；其余文字聊天。"
        return "输入牌面出牌；- 过；/hint 规则提示；/auto on|off 服务器托管。"

    def show_detail(self, title, rows):
        self.detail = title
        document = title + "（PgUp/PgDn 滚动，Esc 关闭）\n\n" + "\n".join(rows)
        self.details.buffer.set_document(self.Document(document, 0), bypass_readonly=True)

    def require_turn(self, phase):
        s = self.client.state
        if not s or s.get("phase") != phase:
            raise ClientError("当前不在" + ("叫分" if phase == "bidding" else "出牌") + "阶段。")
        if s.get("turn") != s.get("seat"):
            raise ClientError("还没有轮到你。")
        return s

    async def send_chat(self, message):
        if not message.strip():
            raise ClientError("聊天内容不能为空。")
        if len(message.strip().encode("utf-16-le")) // 2 > 60:
            raise ClientError("聊天最多 60 个 UTF-16 字符。")
        await self.client.call("chat", text=message.strip())
        self.set_notice("聊天已发送。")

    async def play(self, text, choice_index=None):
        s = self.require_turn("playing")
        rs = parse_ranks(text)
        if rs is None:
            raise ClientError("牌面格式不正确；示例 789XJQKA。")
        cards = select_cards(s.get("hand"), rs)
        readings = classify(cards)
        options = [p for p in readings if beats(p, to_beat(s))]
        if not options:
            raise ClientError("这不是合法牌型。" if not readings else "这手牌管不上。")
        if len(options) > 1 and choice_index is None:
            self.pending = (s["id"], s["version"], text, options)
            summary = "  ".join(f"{i+1}:{describe(p)}" for i, p in enumerate(options))
            self.set_notice(f"请选择牌型 {summary}；/choose 编号 或 {text}#编号。")
            return
        index = 0 if choice_index is None else choice_index - 1
        if not 0 <= index < len(options):
            raise ClientError("牌型编号无效。")
        await self.client.call("play", cards=",".join(map(str, cards)),
                               choice=pattern_key(options[index]))
        self.pending = None
        self.set_notice("已出 " + cards_text(cards))

    async def enter_or_queue(self, mode=None):
        if mode is not None:
            self.client.queue_mode = mode
        status = self.client.lobby_status
        if status.get("queued"):
            raise ClientError("已经在匹配中；/cancel 可取消。")
        if status.get("game"):
            await self.client.load_game(status["game"])
        else:
            await self.client.call("queue")
            if self.client.lobby_status.get("game"):
                await self.client.load_game(self.client.lobby_status["game"])
        self.observe_match()
        self.set_notice("已进入对局。" if self.client.state else "已加入匹配队列。")

    def observe_match(self):
        state = self.client.state
        if between_rounds(state):
            key = (match_of(state).get("id"), state.get("id"))
            if self.break_game != key:
                self.break_game = key
                self.break_deadline = time.monotonic() + INTERMISSION_SECONDS
                self.pending = None
                self.detail = ""
                self.buffer.reset()
                self.set_notice("本局结束；正在显示比赛分表，随后自动换局。")
        else:
            self.break_game = None
            self.break_deadline = 0.0

    async def advance_match(self):
        self.observe_match()
        state = self.client.state
        if not between_rounds(state):
            return False
        m = match_of(state)
        next_game = m.get("next_game")
        if not next_game or time.monotonic() < self.break_deadline:
            return False
        if (not isinstance(next_game, int) or next_game <= 0 or
                next_game == state.get("id")):
            raise ClientError("服务器下一局编号无效，继续等待同步。")
        await self.client.load_game(next_game, expected_match=m["id"],
                                    after_round=m.get("round", 0))
        self.pending = None
        self.detail = ""
        self.scroll = {"log": 0, "chat": 0}
        self.buffer.reset()  # 不能把上一局未发送的输入提交到新局
        self.observe_match()
        new = match_of(self.client.state)
        self.set_notice(f"已进入记分比赛第 {new.get('round','—')}/{new.get('rounds',9)} 局。")
        return True

    async def command(self, submission):
        stamp = submission if isinstance(submission, Submitted) else None
        line = stamp.line if stamp else submission
        cmd, _, arg = line.partition(" ")
        cmd, arg = cmd.lower(), arg.strip()
        state = self.client.state
        match = match_of(state)
        safe_local = ("/quit", "/exit", "/help", "/?", "/close", "/score",
                      "/scores", "/refresh", "/browser")
        if stamp and cmd not in safe_local and (stamp.game_id != (state or {}).get("id") or
                                                 stamp.during_break):
            raise ClientError("输入来自上一局或局间等待，未提交。请根据当前牌局重新输入。")
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
            self.set_notice("已屏蔽他人聊天。" if self.muted else "已恢复聊天。")
            return
        if cmd in ("/score", "/scores"):
            if not match:
                raise ClientError("当前没有记分比赛。")
            if not between_rounds(state):
                self.show_detail("比赛分表", scoreboard_lines(state, self.app.output.get_size().columns))
            return
        if cmd in ("/fair", "/initial", "/log", "/chat", "/hint"):
            if not state:
                raise ClientError("请先进入对局。")
            if cmd == "/fair":
                self.show_detail("发牌公平性", fairness_lines(self.client, True))
            elif cmd == "/log":
                self.show_detail("全部日志", pack_logs(
                    compact_log(state), self.app.output.get_size().columns))
            elif cmd == "/chat":
                self.show_detail("全部聊天", [
                    f"{clean(m.get('username',''))}: {clean(m.get('text',''))}"
                    for _, m in sorted(self.client.chat.items())
                    if not self.muted or m.get("seat") == state.get("seat")])
            elif cmd == "/initial":
                deals = ((state.get("fairness") or {}).get("deals") or [])
                if not deals:
                    raise ClientError("开局牌序在终局后才由服务器公开。")
                deck = deals[-1]["deck"]
                rows = [name_of(state, i) + ": " + cards_text(deck[17*i:17*i+17], True)
                        for i in seat_order(state)]
                rows.append("底牌：" + cards_text(deck[51:]))
                self.show_detail("最后一次发牌的开局手牌", rows)
            else:
                self.require_turn("playing")
                choices = [cards_text(c) for c in hints(state["hand"], to_beat(state))]
                self.set_notice("规则提示：" + (" / ".join(choices[:12]) if choices else "没有能管上的牌。"))
            return
        if self.demo:
            self.set_notice("离线演示仅查看界面，不提交网络请求。")
            return
        if cmd == "/browser":
            async with self.client.transport.lock:
                await self.client.transport.enable_browser()
            if self.client.token:
                await self.client.refresh_token()
            else:
                await self.client.load_lobby()
            await self.client.call("state" if self.client.state else "lobby")
            self.connected = True
            self.observe_match()
            return
        if cmd == "/refresh":
            if self.client.token:
                await self.client.refresh_token()
            else:
                await self.client.load_lobby()
            await self.client.call("state" if self.client.state else "lobby")
            self.connected = True
            self.observe_match()
            self.set_notice("状态已刷新。")
            return
        if not self.connected:
            raise ClientError("尚未连接；可用 /refresh 或 /browser 重试。")
        if cmd in ("/home", "/again"):
            if state and state.get("phase") != "finished":
                raise ClientError("本局尚未结束。")
            if match and not match.get("finished"):
                raise ClientError("九局比赛尚未结束；局间会自动换局。")
            previous_mode = "match" if match else "single"
            if self.client.lobby_status.get("queued") and not state:
                await self.client.call("cancel")
            await self.client.load_lobby()
            await self.client.call("lobby")
            self.detail, self.pending = "", None
            if cmd == "/again":
                await self.enter_or_queue(previous_mode)
            return
        if cmd == "/cancel":
            if state:
                raise ClientError("已进入对局，不能取消匹配。")
            await self.client.call("cancel")
            self.set_notice("匹配已取消。" if not self.client.lobby_status.get("game")
                            else "匹配已成功，请输入 1 返回对局。")
            return
        if not state:
            if line in ("1", "2"):
                await self.enter_or_queue("single" if line == "1" else "match")
            else:
                raise ClientError("大厅请输入 1、2 或 /help。")
            return
        if between_rounds(state):
            raise ClientError("正在等待下一局，不允许提交旧牌或聊天。")
        if cmd == "/say":
            await self.send_chat(arg)
            return
        if cmd == "/auto":
            if state.get("phase") == "finished":
                raise ClientError("对局已经结束。")
            on = arg.lower()
            if on not in ("on", "off", "1", "0"):
                raise ClientError("用法：/auto on 或 /auto off")
            await self.client.call("auto", on="1" if on in ("on", "1") else "0")
            self.set_notice("服务器托管状态已更新。")
            return
        if cmd == "/choose":
            if not arg.isdigit() or not self.pending:
                raise ClientError("没有待选择的牌型或编号不正确。")
            game, version, pending_text, _ = self.pending
            if (game, version) != (state["id"], state["version"]):
                self.pending = None
                raise ClientError("牌局已变化，请重新输入要出的牌。")
            await self.play(pending_text, int(arg))
            return
        if cmd == "/play":
            if not arg:
                raise ClientError("用法：/play 牌面")
            line = arg
        elif line.startswith("/"):
            raise ClientError("未知命令；/help 查看帮助，用 /say 强制聊天。")
        if state.get("phase") == "bidding" and line in ("-", "0", "1", "2", "3"):
            self.require_turn("bidding")
            value = 0 if line == "-" else int(line)
            if (value == 0 and state.get("must_bid")) or (
                    value and value <= state.get("bid", 0)):
                raise ClientError("必须叫更高的分数。" if value else "本轮必须叫分。")
            await self.client.call("bid", value=value)
            self.set_notice("已不叫。" if not value else f"已叫 {value} 分。")
        elif line == "-":
            self.require_turn("playing")
            if state.get("leading"):
                raise ClientError("自由出牌时不能过。")
            await self.client.call("pass")
            self.set_notice("已过。")
        else:
            choice = re.fullmatch(r"(.+)#([1-9]\d*)", line)
            play_text, index = (choice[1], int(choice[2])) if choice else (line, None)
            if parse_ranks(play_text) is not None or cmd == "/play":
                await self.play(play_text, index)
            else:
                await self.send_chat(line)

    async def command_loop(self):
        while True:
            submission = await self.commands.get()
            self.busy = True
            try:
                async with self.op_lock:
                    await self.command(submission)
            except NetworkError as exc:
                self.set_notice(f"{exc} 操作结果不确定，未重发；请 /refresh 核对状态。")
            except ClientError as exc:
                self.set_notice(str(exc))
            except Exception as exc:
                self.set_notice(f"处理失败（{type(exc).__name__}）；请 /refresh 核对页面结构。")
            finally:
                self.busy = False
                self.wakeup.set()
                self.app.invalidate()

    async def poll_once(self):
        was_queued = self.client.queue_may_be_active or bool(self.client.lobby_status.get("queued"))
        await self.client.call("state" if self.client.state else "lobby")
        if not self.client.state and was_queued and self.client.lobby_status.get("game"):
            await self.client.load_game(self.client.lobby_status["game"])
            self.set_notice("匹配成功，已进入对局。")
        self.observe_match()
        await self.advance_match()
        if self.detail == "比赛分表" and match_of(self.client.state):
            new_text = "\n".join(scoreboard_lines(self.client.state,
                                                  self.app.output.get_size().columns))
            if new_text not in self.details.buffer.text:
                self.show_detail("比赛分表", new_text.splitlines())

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
                        self.set_notice("连接已恢复。")
                    failures = 0
                    self.app.invalidate()
            except LoginError as exc:
                self.connected = False
                self.set_notice(str(exc))
            except ClientError as exc:
                failures += 1
                self.set_notice(f"同步失败：{exc}（轮询会重试）")
                if isinstance(exc, ApiError) and exc.retry_after:
                    await asyncio.sleep(exc.retry_after)
            except Exception as exc:
                failures += 1
                self.set_notice(f"状态格式可能发生变化（{type(exc).__name__}），请检查 QOJ 页面。")

    async def connect(self):
        try:
            async with self.op_lock:
                await self.client.load_lobby()
                await self.client.call("lobby")
                if self.client.lobby_status.get("game"):
                    await self.client.load_game(self.client.lobby_status["game"])
                self.connected = True
                self.observe_match()
                self.set_notice("连接成功，请输入操作。")
        except ClientError as exc:
            self.set_notice(f"{exc} 可用 /refresh 或 /browser 重试。")
        except Exception as exc:
            self.set_notice(f"初始化失败（{type(exc).__name__}）；可用 /refresh 重试。")

    async def run(self):
        tasks = []

        def start():
            tasks.append(asyncio.create_task(self.command_loop()))
            if not self.demo:
                tasks.extend((asyncio.create_task(self.connect()),
                              asyncio.create_task(self.poll_loop())))
        try:
            await self.app.run_async(pre_run=start)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def demo_client():
    client = QojClient(None)
    client.lobby = LobbyInfo(username="me", rating="1500",
                             stats={"总积分": "1", "对局": "20", "胜率": "45.0%", "地主胜率": "75.0%"})
    hand = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 53]
    client.accept({"state": {
        "id": 25012, "version": 8, "phase": "playing", "seat": 2, "turn": 2,
        "players": [{"username": "jia", "count": 17, "bid": 0, "auto": False},
                    {"username": "yi", "count": 17, "bid": 0, "auto": False},
                    {"username": "me", "count": len(hand), "bid": 2, "auto": False}],
        "landlord": 2, "bottom": [48, 52, 53], "hand": hand, "hands": None,
        "bid": 2, "multiplier": 1, "bombs": 0, "redeals": 0, "leading": True,
        "last": None, "table": ["pass", "pass", None],
        "log": [{"kind": "bid", "seat": 2, "value": 2}],
        "result": None, "fairness": {"commitments": ["0" * 64]},
    }, "chat": [{"id": 1, "seat": 0, "username": "jia", "text": "大家好"}]})
    return client


async def main_async(args):
    if args.demo:
        await TerminalUI(demo_client(), demo=True).run()
        return
    cookie = parse_cookie(getpass.getpass("请输入 __Host-UOJSESSID（不回显）："))
    transport = Transport(cookie, args.transport, args.browser, args.timeout)
    client = QojClient(transport)
    ui = TerminalUI(client)
    transport.notify = ui.set_notice
    try:
        await ui.run()
    finally:
        # 退出只尝试取消大厅中的排队，不会自动退出/结束已开始的对局。
        if (client.queue_may_be_active or client.lobby_status.get("queued")) and not client.state:
            try:
                await asyncio.wait_for(client.call("cancel"), timeout=args.timeout + 1)
            except Exception:
                print("未能确认取消匹配，请在 QOJ 网页检查队列状态。")
        await transport.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo", action="store_true", help="离线 UI 演示，不需要 Cookie")
    parser.add_argument("--transport", choices=("auto", "http", "browser"), default="auto")
    parser.add_argument("--browser", choices=("auto", "chromium", "chrome", "msedge"), default="auto")
    parser.add_argument("--timeout", type=int, default=12, help="请求超时秒数（3~60）")
    args = parser.parse_args(argv)
    if not 3 <= args.timeout <= 60:
        parser.error("--timeout 必须为 3~60 秒")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        asyncio.run(main_async(args))
    except (KeyboardInterrupt, EOFError):
        pass
    except ModuleNotFoundError as exc:
        print(f"缺少依赖：{exc.name}；请安装 curl_cffi beautifulsoup4 prompt_toolkit。", file=sys.stderr)
        return 1
    except ClientError as exc:
        print(clean(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
