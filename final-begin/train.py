#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""final-v2：全历史模型的离线 CPU 自博弈训练。Ctrl+C 安全保存并退出。"""
from __future__ import annotations
import argparse
from collections import deque
import datetime as dt
import os
from pathlib import Path
import random
import signal
import sys
import threading
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'client'))
from qoj_cli import (Agent, DEFAULT_RULES, Game, Policy, ROLES, SCORE_SCALE,
                     atomic_save, bundle, check_bundle, legal_actions, role_for)

BUCKETS = ('bid',)+ROLES


class TrainingLock:
    """OS-owned lock, automatically released even if the process is killed."""
    def __init__(self, model):
        self.path = Path(model).with_suffix('.train.lock')
        self.file = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('a+b')
        self.file.seek(0)
        if not self.file.read(1):
            self.file.write(b'0'); self.file.flush()
        self.file.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close(); self.file = None
            raise ValueError('已有训练进程正在写这个模型；请先 Ctrl+C 停止它。') from None
        return self

    def __exit__(self, *args):
        if self.file is not None:
            # Keep the tiny file: unlinking a lock races with a second process.
            self.file.close(); self.file = None


def make_batch(rows):
    """Variable-length histories are padded for batching, never cropped."""
    lengths = torch.tensor([len(r['history']) for r in rows], dtype=torch.long)
    histories = torch.nn.utils.rnn.pad_sequence([r['history'].float() for r in rows], batch_first=True)
    states = torch.stack([r['state'] for r in rows]).float()
    actions = torch.stack([r['action'] for r in rows]).float()
    targets = torch.tensor([r['target'] for r in rows], dtype=torch.float32)
    return states, histories, lengths, actions, targets


def train(args, stop):
    torch.set_num_threads(args.threads)
    rng = random.Random(args.seed); torch.manual_seed(args.seed)
    agent = Agent(args.model, args.threads); policy = agent.policy
    if agent.meta.get('rules') != DEFAULT_RULES:
        raise ValueError('模型的规则不是本包默认规则，不能继续训练。')
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-5)
    replay = {role: deque(maxlen=args.buffer) for role in BUCKETS}
    frames = games = updates = 0
    previous_seconds = 0.0
    recent_wins = deque(maxlen=200); recent_losses = deque(maxlen=100)
    if args.checkpoint.is_file() and not args.fresh:
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        check_bundle(saved)
        if saved.get('training_schema') != 1 or saved['meta'].get('rules') != DEFAULT_RULES:
            raise ValueError('续训文件格式或规则不同，请勿混用 final-v1 续训文件。')
        if saved['meta'].get('run_id') != agent.meta.get('run_id'):
            raise ValueError('current.pt 和 training.pt 来自不同训练。请恢复配套文件，或用 --fresh 从当前模型训练。')
        if saved['meta'].get('frames', 0) < agent.meta.get('frames', 0):
            raise ValueError('续训文件比当前模型更旧；请恢复配套 training.pt，或用 --fresh 继续当前模型。')
        policy.load_state_dict(saved['weights'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        for group in optimizer.param_groups: group['lr'] = args.lr
        agent.meta = saved['meta']
        frames, games, updates = (int(saved['meta'][k]) for k in ('frames', 'games', 'updates'))
        previous_seconds = saved['meta'].get('training_seconds', 0.0)
        rng.setstate(saved['rng']); torch.set_rng_state(saved['torch_rng'])
        for role in BUCKETS: replay[role].extend(saved['replay'][role][-args.buffer:])
        recent_wins.extend(saved.get('recent_wins', []))
        print(f'已恢复：{games} 局 / {frames} 次决策 / {updates} 次更新；回放样本 {sum(map(len, replay.values()))}。', flush=True)
        del saved
    elif args.fresh:
        # Fresh means reset optimizer/replay, not reset learned weights.
        frames = int(agent.meta.get('frames', 0)); games = int(agent.meta.get('games', 0))
        updates = int(agent.meta.get('updates', 0)); previous_seconds = agent.meta.get('training_seconds', 0.0)
        print('从当前模型继续；本次不恢复旧优化器和回放。', flush=True)
    else:
        frames = int(agent.meta.get('frames', 0)); games = int(agent.meta.get('games', 0))
        updates = int(agent.meta.get('updates', 0)); previous_seconds = agent.meta.get('training_seconds', 0.0)
        print('没有续训文件；从 client/models/current.pt 建立新训练状态。', flush=True)

    policy.eval()
    start = last_save = last_report = time.monotonic()
    initial_frames, initial_games = frames, games

    def epsilon():
        progress = min(1.0, frames/max(args.epsilon_steps, 1))
        return args.epsilon+(args.epsilon_end-args.epsilon)*progress

    def meta():
        elapsed = time.monotonic()-start
        return {**agent.meta, 'source': 'Custom full-public-history Monte Carlo self-play',
                'rules': dict(DEFAULT_RULES), 'frames': frames, 'games': games, 'updates': updates,
                'training_seconds': previous_seconds+elapsed,
                'parameters': sum(p.numel() for p in policy.parameters()),
                'epsilon': epsilon(), 'loss': float(np.mean(recent_losses)) if recent_losses else None,
                'saved_utc': dt.datetime.now(dt.timezone.utc).isoformat()}

    def report(saved=False):
        elapsed = time.monotonic()-start
        rate = (frames-initial_frames)/max(elapsed, 1e-9)
        loss = f'{np.mean(recent_losses):.4f}' if recent_losses else '等待样本'
        win = f'{100*np.mean(recent_wins):.1f}%' if recent_wins else '—'
        print(f'{"已保存" if saved else "训练中"}：{games} 局 / {frames} 决策 / {updates} 更新'
              f' | {rate:.1f} 决策/s | loss {loss} | ε {epsilon():.3f} | 近局地主胜率 {win}', flush=True)

    def save():
        nonlocal last_save, last_report
        model = bundle(policy, meta())
        checkpoint = {**model, 'training_schema': 1, 'optimizer': optimizer.state_dict(),
                      'rng': rng.getstate(), 'torch_rng': torch.get_rng_state(),
                      'replay': {k: list(v) for k, v in replay.items()}, 'recent_wins': list(recent_wins)}
        # Checkpoint first: an interrupted publication can be recovered on restart.
        atomic_save(checkpoint, args.checkpoint)
        atomic_save(model, args.model)
        last_save = last_report = time.monotonic()
        report(True)

    def maintenance():
        nonlocal last_report
        now = time.monotonic()
        if now-last_save >= args.save_interval: save()
        elif now-last_report >= args.log_interval:
            report(); last_report = now

    def should_stop():
        return stop.is_set() or (args.seconds > 0 and time.monotonic()-start >= args.seconds)

    def learn():
        nonlocal updates
        policy.train()
        try:
            for _ in range(args.updates_per_game):
                eligible = [k for k in BUCKETS if len(replay[k]) >= args.min_samples]
                if not eligible or should_stop(): break
                # Equal bucket weight keeps the relatively rare bidding actions learnable.
                allocation = {k: args.batch//len(eligible) for k in eligible}
                for k in rng.sample(eligible, args.batch%len(eligible)): allocation[k] += 1
                rows = []
                for k, count in allocation.items():
                    pool = list(replay[k])
                    rows += rng.sample(pool, count) if len(pool) >= count else rng.choices(pool, k=count)
                states, histories, lengths, actions, targets = make_batch(rows)
                predicted = policy(states, histories, lengths, actions)
                # True terminal scores: reward = own final points / 6 for every role.
                loss = torch.nn.functional.mse_loss(predicted, targets)
                if not torch.isfinite(loss): raise RuntimeError('训练出现非有限损失，停止；请恢复之前保存的模型。')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step(); updates += 1; recent_losses.append(float(loss.detach()))
                maintenance()
        finally:
            policy.eval()

    save()
    try:
        while not should_stop() and (not args.games or games-initial_games < args.games):
            game = Game(rng.getrandbits(63)); trajectory = []
            for _ in range(256):
                if should_stop(): break
                state = game.observation(); actions = legal_actions(state)
                q, (x, history, candidates) = agent.values(state, actions, return_features=True)
                index = rng.randrange(len(actions)) if rng.random() < epsilon() else int(q.argmax())
                bucket = 'bid' if state['phase'] == 'bidding' else role_for(state['seat'], state['landlord'])
                # Keep replay and computation float32 on both Windows ARM64 and x64.
                row = {'state': torch.from_numpy(x), 'history': torch.from_numpy(history),
                       'action': torch.from_numpy(candidates[index].copy())}
                trajectory.append((bucket, state['seat'], row))
                game.apply(actions[index]); frames += 1
                if game.phase == 'finished':
                    for role, seat, sample in trajectory:
                        sample['target'] = float(game.result['deltas'][seat])/SCORE_SCALE
                        replay[role].append(sample)
                    games += 1; recent_wins.append(int(game.result['landlord_won']))
                    learn(); break
                maintenance()
            else:
                raise RuntimeError('单局超过 256 次动作；已停止，请检查规则。')
            maintenance()
        # Incomplete game's samples are deliberately discarded on interruption.
        save()
    except Exception:
        # Do not overwrite the last valid checkpoint after unexpected failures.
        print('训练异常退出；保留上一次成功保存的模型和续训文件。', file=sys.stderr, flush=True)
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--threads', type=int, default=4, help='CPU 线程数，默认 4')
    p.add_argument('--batch', type=int, default=64, help='每次更新样本数')
    p.add_argument('--buffer', type=int, default=1024, help='叫分及各角色分别保留的回放条数')
    p.add_argument('--min-samples', type=int, default=32, help='某类积累到此样本数后参与训练')
    p.add_argument('--updates-per-game', type=int, default=4)
    p.add_argument('--lr', type=float, default=0.0003)
    p.add_argument('--epsilon', type=float, default=0.20, help='初始随机探索率')
    p.add_argument('--epsilon-end', type=float, default=0.05)
    p.add_argument('--epsilon-steps', type=int, default=200000, help='探索率线性变化所需累计决策数')
    p.add_argument('--save-interval', type=float, default=60, help='保存模型及完整续训文件的秒数间隔')
    p.add_argument('--log-interval', type=float, default=10)
    p.add_argument('--games', type=int, default=0, help='本次完成局数，0 为不限')
    p.add_argument('--seconds', type=float, default=0, help='本次秒数上限，0 为不限')
    p.add_argument('--seed', type=int, default=20261004)
    p.add_argument('--fresh', action='store_true', help='从当前权重继续，忽略旧优化器及回放；不是重置随机权重')
    p.add_argument('--model', type=Path, default=ROOT/'client/models/current.pt')
    p.add_argument('--checkpoint', type=Path, default=ROOT/'training.pt')
    args = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'): stream.reconfigure(encoding='utf8', errors='replace')
    if min(args.threads, args.batch, args.buffer, args.min_samples, args.updates_per_game, args.epsilon_steps) < 1:
        p.error('线程数、批量、容量、最少样本、更新次数和探索步数必须为正。')
    if args.min_samples > args.buffer: p.error('--min-samples 不能大于 --buffer。')
    if min(args.lr, args.save_interval, args.log_interval) <= 0: p.error('学习率、保存和显示间隔必须为正。')
    if not all(0 <= x <= 1 for x in (args.epsilon, args.epsilon_end)): p.error('探索率必须在 0–1 之间。')
    if min(args.games, args.seconds) < 0: p.error('局数、秒数不能为负。')
    args.model = args.model.resolve(); args.checkpoint = args.checkpoint.resolve()
    if args.model == args.checkpoint: p.error('模型路径和续训路径不能相同。')
    if not args.model.is_file(): p.error('缺少初始模型，请保留随包提供的 client/models/current.pt。')
    stop = threading.Event()

    def request_stop(signum, frame):
        if not stop.is_set():
            stop.set()
            print('\n收到停止请求；完成当前计算并保存，请等待“训练已停止”。', flush=True)

    previous = signal.signal(signal.SIGINT, request_stop)
    try:
        with TrainingLock(args.model): train(args, stop)
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(f'训练未完成：{error}', file=sys.stderr, flush=True)
        return 1
    finally:
        signal.signal(signal.SIGINT, previous)
    print('训练已停止，最新模型已保存。可运行 python local.py 或 python client/qoj_cli.py。', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
