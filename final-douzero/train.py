#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""纯 CPU 自博弈训练、恢复和模型发布；不打开对战 UI，不联网。"""
from __future__ import annotations
import argparse
from collections import deque
import datetime as dt
import hashlib
import json
from pathlib import Path
import random
import sys
import threading
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'client'))
from qoj_cli import (Action, Agent, DEFAULT_RULES, Game, Policy, ROLES,
                     atomic_save, bundle, legal_actions, load_bundle)

def record(state, actions, index, features):
    role, z, x, extra = features
    return {'role': role, 'seat': state['seat'], 'z': z[0].astype('int8'),
            'x': x[index].astype('float32' if role == 'bid' else 'int8'),
            'extra': extra[index].copy()}

def trainer(config, stop):
    # Training uses only local simulation and model data; no network requests.
    import numpy as np
    import torch
    torch.set_num_threads(config['threads'])
    rng = random.Random(config['seed']); torch.manual_seed(config['seed'])
    output = Path(config['output']); output.mkdir(parents=True, exist_ok=True)
    deploy = Path(config['deploy_model'])
    agent = Agent(deploy, config['threads']); policy = agent.policy
    optimizer = torch.optim.Adam(policy.optimizer_groups(config['base_lr'], config['new_lr']))
    frames = games = updates = human_games = 0
    elapsed_before = 0.0; losses = {}
    if (output/'latest.pt').exists() and not config['fresh']:
        saved = torch.load(output/'latest.pt', map_location='cpu', weights_only=True)
        if saved['meta']['rules'] != config['rules']:
            raise ValueError('恢复模型的规则不同；请使用不同 --output 或 --fresh。')
        policy.load_state_dict(saved['weights'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        frames, games, updates = (saved['meta'][k] for k in ('frames', 'games', 'updates'))
        human_games = saved['meta'].get('human_games', 0)
        elapsed_before = saved['meta'].get('training_seconds', 0.0)
        rng.setstate(saved['rng']); torch.set_rng_state(saved['torch_rng'])
        print(f'已恢复：{games} 局，{frames} 次决策。', flush=True)
    if not (output/'reference.pt').exists(): atomic_save(bundle(policy, {**agent.meta, 'rules': config['rules']}), output/'reference.pt')
    replay = {role: deque(maxlen=config['buffer']) for role in policy.models}
    start = time.monotonic(); last_live = last_publish = start
    last_frames, last_games = frames, games

    def meta():
        elapsed = time.monotonic()-start
        return {**agent.meta, 'source': 'QOJ additive-score Monte Carlo self-play fine-tuning',
                'rules': config['rules'], 'frames': frames, 'games': games, 'updates': updates,
                'human_games': human_games, 'training_seconds': elapsed_before+elapsed,
                'session_seconds': elapsed, 'session_frames': frames-last_frames,
                'session_games': games-last_games, 'decisions_per_second': (frames-last_frames)/max(elapsed, 1e-9),
                'loss': losses, 'saved_utc': dt.datetime.now(dt.timezone.utc).isoformat()}

    def save(publish=False):
        nonlocal last_live, last_publish
        now = time.monotonic(); m = meta(); model = bundle(policy, m)
        atomic_save(model, output/'live.pt'); last_live = now
        if publish:
            checkpoint = {**model, 'optimizer': optimizer.state_dict(), 'rng': rng.getstate(), 'torch_rng': torch.get_rng_state()}
            atomic_save(checkpoint, output/'latest.pt')
            stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
            atomic_save(checkpoint, output/f'checkpoint_{stamp}_{frames}.pt')
            atomic_save(model, deploy); last_publish = now
            for old in sorted(output.glob('checkpoint_*.pt'))[:-config['keep']]: old.unlink()
            print(f'已保存并发布：{games} 局 / {frames} 次决策 / {updates} 次更新，{m["decisions_per_second"]:.1f} 决策/s。', flush=True)
        else:
            print(f'训练中：{games} 局 / {frames} 次决策 / {updates} 次更新，{m["decisions_per_second"]:.1f} 决策/s。', flush=True)
        temporary = output/'status.writing.json'
        temporary.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding='utf8')
        temporary.replace(output/'status.json')

    def maybe_save():
        request = output/'save.request'
        if request.exists():
            request.unlink(); save(True)
        elif time.monotonic()-last_publish >= config['save_interval']: save(True)
        elif time.monotonic()-last_live >= config['live_interval']: save(False)

    def ingest(trajectory, deltas, landlord):
        for row in trajectory:
            # Bidding has no role yet; learn the acting seat's final payoff.
            scale = 6.0 if row['seat'] == landlord else 3.0
            row['target'] = float(deltas[row['seat']])/scale
            replay[row['role']].append(row)

    def learn():
        nonlocal updates
        policy.train()
        for role, rows in replay.items():
            if len(rows) < config['min_samples']: continue
            batch = rng.sample(list(rows), min(config['batch'], len(rows)))
            z = torch.from_numpy(np.stack([r['z'] for r in batch]).astype(np.float32))
            x = torch.from_numpy(np.stack([r['x'] for r in batch]).astype(np.float32))
            e = torch.from_numpy(np.stack([r['extra'] for r in batch]).astype(np.float32))
            y = torch.tensor([r['target'] for r in batch], dtype=torch.float32)
            pred = policy(role, z, x, e).squeeze(-1)
            loss = torch.nn.functional.mse_loss(pred, y)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.models[role].parameters(), 5.0)
            optimizer.step(); updates += 1; losses[role] = float(loss.detach())
        policy.eval()

    save(True)
    try:
        while not stop.is_set():
            if config['games'] and games-last_games >= config['games']: break
            if config['seconds'] and time.monotonic()-start >= config['seconds']: break
            g = Game(rng.getrandbits(63), config['rules']); trajectory = []
            for _ in range(800):
                maybe_save()
                if stop.is_set() or (config['seconds'] and time.monotonic()-start >= config['seconds']): break
                state = g.observation(); actions = legal_actions(state)
                q, features = agent.values(state, actions, return_features=True)
                index = rng.randrange(len(actions)) if rng.random() < config['epsilon'] else int(q.argmax())
                trajectory.append(record(state, actions, index, features))
                g.apply(actions[index]); frames += 1
                if g.phase == 'finished':
                    ingest(trajectory, g.result['deltas'], g.landlord); games += 1
                    learn(); break
            else: raise RuntimeError('一局超过 800 次动作，可能存在规则错误。')
            maybe_save()
            if config['seconds'] and time.monotonic()-start >= config['seconds']: break
    finally:
        save(True); stop.set()

def import_weights(checkout, output):
    policy = Policy(); hashes = {}
    for role in ROLES:
        path = checkout/'baseline'/'test'/f'{role}.ckpt'
        policy.models[role].base.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        hashes[str(path.relative_to(checkout))] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = checkout/'baseline'/'SLModel'/'bid_weights_new.pkl'
    policy.models['bid'].strength.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    hashes[str(path.relative_to(checkout))] = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = {'source': 'DouZero-ADP play weights + AlphaDou supervised threshold bid; QOJ residuals untrained',
            'frames': 0, 'games': 0, 'updates': 0, 'rules': DEFAULT_RULES, 'source_sha256': hashes,
            'parameters': sum(p.numel() for p in policy.parameters()),
            'trainable_parameters': sum(p.numel() for p in policy.parameters() if p.requires_grad)}
    atomic_save(bundle(policy, meta), output)
    print(meta)

def main(argv=None):
    p = argparse.ArgumentParser(description='仅进行本地 CPU 自博弈微调；人机对战请运行 local.py。')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--buffer', type=int, default=2000, help='每个角色的回放容量')
    p.add_argument('--min-samples', type=int, default=32)
    p.add_argument('--epsilon', type=float, default=0.05)
    p.add_argument('--base-lr', type=float, default=1e-5)
    p.add_argument('--new-lr', type=float, default=1e-4)
    p.add_argument('--save-interval', type=float, default=600)
    p.add_argument('--live-interval', type=float, default=30)
    p.add_argument('--keep', type=int, default=6)
    p.add_argument('--games', type=int, default=0, help='本次完成多少局后停止；0 为不限')
    p.add_argument('--seconds', type=float, default=0, help='本次训练秒数上限；0 为不限')
    p.add_argument('--force-after-redeals', type=int, default=3, help='默认流局 3 次后下一次发牌强制最后一家叫分')
    p.add_argument('--redeal-start', choices=('random', 'same'), default='random')
    p.add_argument('--seed', type=int, default=20261004)
    p.add_argument('--fresh', action='store_true', help='忽略 latest.pt，从当前 client 模型开始')
    p.add_argument('--output', type=Path, default=ROOT/'train/checkpoints')
    p.add_argument('--deploy-model', type=Path, default=ROOT/'client/models/current.pt')
    p.add_argument('--publish', type=Path, help='仅向客户端发布一个已有模型或快照，不训练')
    p.add_argument('--import-pretrained', type=Path, metavar='CHECKOUT', help='仅从本地 AlphaDou 源码导入初始模型')
    args = p.parse_args(argv)
    if args.publish and args.import_pretrained: p.error('--publish 和 --import-pretrained 只能选一个。')
    if args.publish:
        policy, data = load_bundle(args.publish)
        atomic_save(bundle(policy, data['meta']), args.deploy_model)
        print('已发布模型：'+str(args.deploy_model.resolve()))
        return 0
    if args.import_pretrained:
        import_weights(args.import_pretrained, ROOT/'client/models/imported.pt')
        return 0
    if min(args.threads, args.batch, args.buffer, args.min_samples, args.keep) < 1: p.error('线程数、批量、容量、最少样本和保留数必须为正整数。')
    if args.min_samples > args.buffer: p.error('--min-samples 不能大于 --buffer，否则永远不会更新模型。')
    if min(args.save_interval, args.live_interval, args.base_lr, args.new_lr) <= 0: p.error('保存间隔和学习率必须大于零。')
    if not 0 <= args.epsilon <= 1 or min(args.games, args.seconds, args.force_after_redeals) < 0: p.error('探索率为 0–1，局数、秒数和流局次数为非负。')
    config = dict(vars(args)); config['output'] = str(args.output.resolve()); config['deploy_model'] = str(args.deploy_model.resolve())
    config['rules'] = {**DEFAULT_RULES, 'force_after_redeals': args.force_after_redeals, 'redeal_start': args.redeal_start}
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'): stream.reconfigure(encoding='utf8', errors='replace')
    if not args.deploy_model.exists(): p.error('部署模型不存在；请保留随包提供的 client/models/current.pt。')
    stop = threading.Event()
    try:
        trainer(config, stop)
    except KeyboardInterrupt:
        print('训练已停止，模型已保存。', flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
