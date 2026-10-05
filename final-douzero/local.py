#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一人两人机，本地全屏斗地主；打一局后自动退出并打印完整日志。"""
from __future__ import annotations
import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'client'))
import qoj_cli as core

LOCAL_HELP = '叫分 0/1/2/3；牌面出牌，- 过；/ai 提示；/bot on|off；/sort；/play 牌面；/choose 编号；/quit。F2 翻日志，Esc 回最新。'


def model_path(explicit=None):
    if explicit is not None:
        return Path(explicit)
    paths = [core.MODEL_PATH, ROOT/'train/checkpoints/live.pt']
    available = []
    for path in paths:
        try:
            available.append((path.stat().st_mtime_ns, path))
        except FileNotFoundError:
            pass
    if not available:
        raise core.ClientError('模型不存在，请保留 client/models/current.pt。')
    return max(available, key=lambda row: row[0])[1]


class LocalClient:
    """Local state adapter: expose only the human hand and public actions."""
    def __init__(self, seat=0, seed=None, rules=None):
        self.me = seat
        self.game = core.Game(seed, rules)
        self.names = ['人机A', '人机B', '人机C']
        self.names[seat] = '你'
        for index, other in enumerate(i for i in range(3) if i != seat):
            self.names[other] = '人机'+chr(ord('A')+index)
        self.auto = False
        self.history = []
        self.state = None
        self.on_change = None
        self.sync()
        self.remember_deal()

    def remember_deal(self):
        g = self.game
        self.history += [f'发牌 #{g.redeals+1}；先叫：{self.names[g.first]}。',
                         '你的开局手牌：'+core.cards_text(g.hands[self.me], True)]

    def sync(self):
        s = self.game.observation(self.me)
        for seat, player in enumerate(s['players']):
            player['username'] = self.names[seat]
            player['auto'] = seat != self.me or self.auto
        table = [None]*3
        last_seat = None
        for event in core.current_log(s):
            if event.get('kind') == 'play':
                if last_seat is None or last_seat == event['seat']:
                    table = [None]*3
                table[event['seat']] = {'cards':event['cards'], 'pattern':event['pattern']}
                last_seat = event['seat']
            elif event.get('kind') == 'pass':
                table[event['seat']] = 'pass'
        s['table'] = table
        self.state = s

    def set_auto(self, enabled):
        if self.auto == enabled:
            return
        self.auto = enabled
        self.history.append('你的本地托管：'+('开启' if enabled else '关闭'))
        self.sync()

    def apply(self, action, automatic=False):
        g = self.game
        start = len(g.log)
        forced = action.kind == 'bid' and g.must_bid
        g.apply(action)
        for event in g.log[start:]:
            if event['kind'] in ('bid','play','pass'):
                event['auto'] = automatic
            name = self.names[event['seat']] if 'seat' in event else ''
            mark = ' [托管]' if event.get('auto') else ''
            if event['kind'] == 'bid':
                text = f'叫 {event["value"]} 分' if event['value'] else '不叫'
                self.history.append(name+mark+'：'+text+('（强制叫分）' if forced else ''))
            elif event['kind'] == 'landlord':
                self.history.append(f'地主：{name}；底分 {event["value"]}；公开底牌：'+core.cards_text(event['cards']))
            elif event['kind'] == 'play':
                self.history.append(name+mark+'：'+core.cards_text(event['cards'])+' · '+core.describe(event['pattern'])
                                    +f'；剩 {len(g.hands[event["seat"]])} 张')
            elif event['kind'] == 'pass':
                self.history.append(name+mark+'：过')
            elif event['kind'] == 'redeal':
                self.history.append(f'三家均不叫，重新发牌；累计流局 {g.redeals} 次。')
                self.remember_deal()
            elif event['kind'] == 'finish':
                self.history.append(self.result_text())
        self.sync()
        if self.on_change is not None:
            self.on_change()

    def result_text(self):
        r = self.game.result
        if not r:
            return '本局提前退出，未结算。'
        tag = '；春天' if r['spring'] else '；反春' if r['anti_spring'] else ''
        scores = ' / '.join(f'{self.names[i]} {delta:+}' for i, delta in enumerate(r['deltas']))
        return f'结算：{"地主胜" if r["landlord_won"] else "农民胜"}{tag}；底分 {r["base"]} × 倍率 {r["multiplier"]}；'+scores

    async def call(self, action, **extra):
        if action == 'state':
            self.sync()
            return {'state':self.state}
        if self.game.turn != self.me or self.game.phase == 'finished':
            raise core.ClientError('还没有轮到你。')
        if self.auto:
            raise core.ClientError('正在托管；先 /bot off，再手动操作。')
        if action == 'bid':
            candidate = core.Action('bid', value=int(extra['value']))
        elif action == 'pass':
            candidate = core.Action('pass')
        elif action == 'play':
            cards = tuple(map(int, extra['cards'].split(',')))
            kind, rank, length = extra['choice'].split(':')
            candidate = core.Action('play', tuple(sorted(map(core.rank, cards))), (kind,int(rank),int(length)))
        else:
            raise core.ClientError('本地对战不支持这个操作。')
        core.validate_action(self.state, candidate)
        self.apply(candidate)
        return {'state':self.state}

    def print_history(self):
        print('\n本地斗地主 · 完整日志'+('（已结束）' if self.game.result else '（提前退出）'))
        for index, line in enumerate(self.history, 1):
            print(f'{index:03d}  {line}')
        if not self.game.result:
            print(self.result_text())


@dataclass(frozen=True)
class Submitted:
    line: str
    version: int


class LocalInputQueue(asyncio.Queue):
    def __init__(self, client):
        super().__init__(maxsize=1)
        self.client = client

    def put_nowait(self, line):
        if isinstance(line, str):
            line = Submitted(line, self.client.game.version)
        return super().put_nowait(line)


class LocalUI(core.BaseTerminalUI):
    def __init__(self, client, model=None, input=None, output=None):
        super().__init__(client, input=input, output=output, show_chat=False)
        self.connected = True
        self.commands = LocalInputQueue(client)
        self.model = model
        self.ai_engine = None
        self.engine_lock = threading.Lock()
        self.ended = asyncio.Event()
        self.failure = None
        client.on_change = self.changed
        self.notice = '已开局；对手手牌隐藏。/ai 提示，/bot on 托管，/help 查看操作。'

    def engine(self):
        path = model_path(self.model)
        if self.ai_engine is not None and self.ai_engine.path != str(path):
            self.ai_engine.close()
            self.ai_engine = None
        if self.ai_engine is None:
            self.ai_engine = core.BoundedEngine(path)
        return self.ai_engine

    def choose(self, state):
        """All loading and switching runs in a thread, within one time budget."""
        started = time.monotonic()
        deadline = started+4.0
        result = {'action':core.minimal_action(state).to_dict(), 'suggestions':[],
                  'fallback':True, 'retry':True, 'status':'busy', 'reason':'本地引擎正在处理上一条请求。'}
        acquired = self.engine_lock.acquire(timeout=max(0, deadline-time.monotonic()))
        try:
            if acquired and not self.quitting:
                engine = self.engine()
                engine.budget = max(0.001, deadline-time.monotonic())
                result = engine.choose(state)
        except (core.ClientError, OSError, ValueError) as error:
            result.update(retry=False, status='error', reason='本地引擎错误：'+str(error))
        finally:
            if acquired:
                self.engine_lock.release()
            result['elapsed'] = time.monotonic()-started
        return result

    def close_engine(self):
        with self.engine_lock:
            if self.ai_engine is not None:
                self.ai_engine.close()
                self.ai_engine = None

    def changed(self):
        self.pending = None
        if self.client.game.phase == 'finished':
            self.ended.set()
        self.app.invalidate()

    def log_height(self):
        return max(2, self.app.output.get_size().rows-21)

    def render_header(self):
        owner = '你的托管：开' if self.client.auto else '你的托管：关'
        return ' 本地斗地主 | 一人两人机 | 完全离线 | '+owner+' | /help'

    def player_row(self, seat, mine=False):
        return super().player_row(seat, mine).replace(' · 叫-1分', ' · 未叫')

    def render_board(self):
        s = self.client.state
        phase = {'bidding':'叫分中','playing':'出牌中','finished':'已结束'}[s['phase']]
        who = self.client.names[s['turn']]
        turn = f'轮到 {who}' if s['phase'] != 'finished' else '本局结束，即将输出完整日志'
        rows = [f'{phase} · {turn} · 底分 {s["bid"]} · ×{s["multiplier"]} · 炸弹 {s["bombs"]} · 流局 {s["redeals"]}',
                '底牌：'+(core.cards_text(s['bottom']) if s['landlord'] is not None else '未公开')
                +(' · 自由出牌' if s['phase']=='playing' and s['leading'] else ''),
                '记牌器（未出现，不含我的手牌）：'+core.counter_text(s)]
        for seat in core.seat_order(s)[:-1]:
            rows.append(self.player_row(seat))
        if s['must_bid']:
            rows.append('本轮最后一家必须叫分。')
        if s['result']:
            rows.append(self.client.result_text())
        return '\n'.join(rows)

    def render_prompt(self):
        s = self.client.state
        if s['phase'] == 'finished':
            return '即将自动退出并打印完整日志。'
        if s['turn'] != self.client.me:
            return self.client.names[s['turn']]+' 正在行动；可 /bot on|off 或 /sort。'
        if self.client.auto:
            return '你的模型托管中；/bot off 接管。'
        if s['phase'] == 'bidding':
            minimum = s['bid']+1
            return ('必须叫分：' if s['must_bid'] else '0 / - 不叫；')+f'{minimum}–3 叫分；/ai 提示。'
        return '输入牌面出牌（X=10 S=小王 D=大王）；- 过；/ai 提示；/bot on 托管。'

    async def command(self, submitted):
        line = submitted.line if isinstance(submitted, Submitted) else submitted
        cmd, _, arg = line.strip().partition(' ')
        cmd, arg = cmd.lower(), arg.strip()
        if cmd in ('/quit','/exit'):
            self.quitting = True
            self.app.exit()
            return
        if cmd in ('/help','/?'):
            self.set_notice(LOCAL_HELP)
            return
        if cmd == '/sort':
            self.desc = not self.desc
            return
        if cmd == '/bot':
            if arg.lower() not in ('on','off'):
                raise core.ClientError('用法：/bot on 或 /bot off')
            self.client.set_auto(arg.lower() == 'on')
            self.set_notice('你的本地托管：'+('开启' if self.client.auto else '关闭'))
            return
        if isinstance(submitted, Submitted) and submitted.version != self.client.game.version:
            raise core.ClientError('牌局已变化，这条输入未执行；请重新输入。')
        s = self.client.state
        if s['phase'] == 'finished' or s['turn'] != self.client.me:
            raise core.ClientError('请在轮到你时输入叫分、牌面或 /ai。')
        if self.client.auto:
            raise core.ClientError('正在托管；先 /bot off，再手动操作。')
        if cmd == '/ai':
            if arg:
                raise core.ClientError('用法：/ai')
            version = s['version']
            result = await asyncio.to_thread(self.choose, core.public_snapshot(s))
            if self.client.game.version != version:
                self.set_notice('牌局已变化，旧提示已丢弃。')
                return
            if result['fallback']:
                self.set_notice('未生成模型提示：'+result['reason']+f'；{result["elapsed"]:.3f}s')
                return
            action = core.Action.from_dict(result['action'])
            estimate = f'；估值 {result["suggestions"][0]["value"]:+.3f}（非胜率）'
            message = '模型建议：'+core.action_text(action)+estimate+f'；{result["elapsed"]:.3f}s'
            self.set_notice(message)
            return
        if cmd == '/choose':
            if not arg.isdigit() or not self.pending:
                raise core.ClientError('当前没有待选择的牌型，或编号无效。')
            game, version, text, _ = self.pending
            if version != s['version']:
                self.pending = None
                raise core.ClientError('牌局已变化，请重新输入。')
            await self.play(text, int(arg))
            return
        if cmd == '/play':
            line = arg
        elif cmd.startswith('/'):
            raise core.ClientError('本地对战不支持此命令；/help 查看可用命令。')
        if s['phase'] == 'bidding':
            if line not in ('-','0','1','2','3'):
                raise core.ClientError('叫分请输入 0/1/2/3，或 -。')
            await self.client.call('bid', value=0 if line=='-' else int(line))
            self.set_notice('你：不叫' if line in ('-','0') else f'你：叫 {line} 分')
        elif line == '-':
            await self.client.call('pass')
            self.set_notice('你：过')
        else:
            text, sep, choice = line.partition('#')
            if core.parse_ranks(text) is None:
                raise core.ClientError('本地对战没有聊天；请输入牌面或 /help。')
            if sep and (not choice.isdigit() or int(choice)<1):
                raise core.ClientError('牌型编号无效。')
            await self.play(text, int(choice) if sep else None)

    async def command_loop(self):
        while True:
            submitted = await self.commands.get()
            self.busy = True
            try:
                async with self.op_lock:
                    await self.command(submitted)
            except (core.ClientError, ValueError, TypeError) as error:
                self.set_notice(str(error))
            finally:
                self.busy = False
                self.app.invalidate()

    async def bot_loop(self):
        while not self.quitting and not self.ended.is_set():
            g = self.client.game
            if self.busy or (g.turn==self.client.me and not self.client.auto):
                await asyncio.sleep(0.05)
                continue
            version, seat = g.version, g.turn
            state = core.public_snapshot(g.observation())
            self.set_notice(self.client.names[seat]+' 正在思考…')
            result = await asyncio.to_thread(self.choose, state)
            if self.quitting:
                break
            async with self.op_lock:
                if g.version != version or g.turn != seat or (seat==self.client.me and not self.client.auto):
                    continue
                if result['fallback']:
                    if not result.get('retry'):
                        raise core.ClientError('本地模型无法接管：'+result['reason'])
                    self.set_notice(self.client.names[seat]+' 等待模型：'+result['reason'])
                else:
                    action = core.Action.from_dict(result['action'])
                    self.client.apply(action, automatic=True)
                    self.set_notice(self.client.names[seat]+'：'+core.action_text(action))
            await asyncio.sleep(0.15 if result['fallback'] else 0.30)

    async def finish_loop(self):
        await self.ended.wait()
        await asyncio.sleep(0.50)
        if self.app.is_running:
            self.app.exit()

    async def guarded(self, coroutine):
        try:
            await coroutine
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.failure = f'{type(error).__name__}: {error}'
            self.client.history.append('本地程序中断：'+self.failure)
            if self.app.is_running:
                self.app.exit()

    async def run(self):
        tasks = []
        def start():
            for work in (self.command_loop(), self.bot_loop(), self.finish_loop()):
                tasks.append(asyncio.create_task(self.guarded(work)))
        try:
            await self.app.run_async(pre_run=start)
        except EOFError:
            self.quitting = True
        finally:
            self.quitting = True
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.to_thread(self.close_engine)


async def run_game(args):
    path = model_path(args.model)
    _, data = core.load_bundle(path)
    client = LocalClient(args.seat-1, args.seed, data['meta'].get('rules', core.DEFAULT_RULES))
    ui = LocalUI(client, args.model)
    try:
        await ui.run()
    finally:
        client.print_history()
    return 1 if ui.failure else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, help='固定模型路径；默认自动使用最新的 current.pt/live.pt')
    parser.add_argument('--seat', type=int, choices=(1,2,3), default=1, help='你的座位，默认 1')
    parser.add_argument('--seed', type=int, help='可选复现种子；默认随机发牌和先叫者')
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf8', errors='replace')
    try:
        return asyncio.run(run_game(args))
    except (KeyboardInterrupt, EOFError):
        return 0
    except (core.ClientError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    core.mp.freeze_support()
    raise SystemExit(main())
