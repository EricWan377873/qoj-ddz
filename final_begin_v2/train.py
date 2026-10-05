#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""final-v3-f6：四角色约 4M 参数模型的自博弈训练；Surface ARM64 自动尝试 Hexagon NPU rollout。"""
from __future__ import annotations

import argparse
import copy
from collections import deque
import datetime as dt
import json
import os
from pathlib import Path
import platform
import random
import signal
import sys
import threading
import time
import uuid
import warnings

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'client'))
from qoj_cli import (  # noqa: E402
    ACTION_DIM, ARCHITECTURE, DEFAULT_RULES, EVENT_DIM, Game, MAX_HISTORY,
    MODEL_PARTS, ROLES, Policy, SCORE_SCALE, STATE_DIM, Agent, atomic_save, bundle,
    check_bundle, legal_actions, model_part_for, prepare, role_for,
)

BUCKETS = MODEL_PARTS
TRAINING_SCHEMA = 5
NPU_ACTION_BATCH_DEFAULT = 16
NPU_HISTORY_DEFAULT = 128


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
            self.file.close(); self.file = None


def cpu_threads(requested):
    return requested if requested > 0 else max(1, os.cpu_count() or 4)


def initial_meta(policy, seed):
    counts = policy.parameter_counts()
    return {
        'source': 'QOJ final-v3-f6 four-role self-play',
        'rules': dict(DEFAULT_RULES),
        'run_id': uuid.uuid4().hex,
        'seed': int(seed),
        'frames': 0, 'games': 0, 'updates': 0,
        'updates_by_part': {k: 0 for k in BUCKETS},
        'training_seconds': 0.0,
        'parameters': sum(counts.values()),
        'parameters_by_part': counts,
        'architecture': dict(ARCHITECTURE),
        'saved_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def ensure_model(model_path, checkpoint_path, seed):
    """Create the ~4M random model on first training run; recover from checkpoint if possible."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.is_file():
        return False
    if checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        check_bundle(saved)
        public = {k: saved[k] for k in ('schema', 'architecture', 'weights', 'meta')}
        atomic_save(public, model_path)
        print('current.pt 缺失：已从 training.pt 恢复模型权重。', flush=True)
        return True
    torch.manual_seed(seed)
    policy = Policy()
    atomic_save(bundle(policy, initial_meta(policy, seed)), model_path)
    counts = policy.parameter_counts()
    print('未发现模型：已自动创建 final-v3-f6 随机初始模型。', flush=True)
    print('参数：' + ' / '.join(f'{k} {v:,}' for k, v in counts.items())
          + f'；总计 {sum(counts.values()):,}。', flush=True)
    return True


def make_batch(rows):
    lengths = torch.tensor([len(r['history']) for r in rows], dtype=torch.long)
    histories = torch.nn.utils.rnn.pad_sequence(
        [r['history'].float() for r in rows], batch_first=True)
    states = torch.stack([r['state'] for r in rows]).float()
    actions = torch.stack([r['action'] for r in rows]).float()
    targets = torch.tensor([r['target'] for r in rows], dtype=torch.float32)
    return states, histories, lengths, actions, targets


class _FixedQnnActor(torch.nn.Module):
    """Static-shape, float-only wrapper for one role network on Hexagon."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, state, history, valid_mask, last_mask, inv_length, actions):
        encoded = self.net.encode_fixed(state, history, valid_mask, last_mask, inv_length)
        # Do NOT use Tensor.expand here. Legacy ONNX lowers fixed expand through
        # ConstantOfShape/Equal/Where, which can force QNN HTP to partition the graph.
        # A zero-valued action slice gives ONNX a fixed [ACTION_BATCH,1] tensor and
        # relies only on ordinary Add broadcasting (native QNN op).
        encoded = encoded + actions[:, :1] * 0.0
        return self.net.score(encoded, actions)


class NpuStepFallback(RuntimeError):
    """A single state is outside the fast fixed-shape NPU envelope.

    This is not a fatal NPU error.  The caller should evaluate only this state
    with the CPU policy and keep the NPU actor alive for later states.
    """


class QnnRollout:
    """Fixed-shape ONNX/QNN actor for Windows ARM64 Snapdragon NPU.

    The model weights are identical to the CPU learner.  Only the rollout wrapper
    uses smaller fixed envelopes (history/action padding) so HTP does less useless
    work.  States outside the history envelope fall back to CPU for that one step.
    """
    def __init__(self, policy, cache_dir, *, action_batch=16, history_steps=128,
                 quiet=True, perf_mode='burst', rpc_latency=100):
        machine = platform.machine().lower()
        if platform.system() != 'Windows' or machine not in ('arm64', 'aarch64'):
            raise RuntimeError('QNN NPU 自动模式只在 Windows ARM64 上启用。')
        try:
            import onnx
            import onnxruntime as ort
            import onnxruntime_qnn as qnn_ep
        except ImportError as e:
            raise RuntimeError('缺少 NPU 依赖；请重新执行 pip install -r client/requirements.txt。') from e
        self.onnx, self.ort, self.qnn_ep = onnx, ort, qnn_ep
        # The periodic session rebuild warnings in f4 are emitted through ORT's
        # default logger, not Python warnings. Silence WARNING globally for this
        # process unless explicit NPU diagnostics were requested; ERROR remains.
        set_log = getattr(ort, 'set_default_logger_severity', None)
        if set_log is not None:
            set_log(2 if not quiet else 3)  # 2=WARNING, 3=ERROR
        ort_ver = getattr(ort, '__version__', '?')
        qnn_ver = getattr(qnn_ep, '__version__', '?')
        if ort_ver != '1.27.0' or qnn_ver != '2.6.0':
            raise RuntimeError(
                f'NPU 运行时版本不匹配：onnxruntime={ort_ver}, onnxruntime-qnn={qnn_ver}。'
                ' final-v3-f5 固定使用官方测试组合 1.27.0 + 2.6.0；请按 README 先卸载再重装。'
            )
        self.action_batch = int(action_batch)
        self.history_steps = int(history_steps)
        if self.action_batch < 1:
            raise RuntimeError('NPU action batch 必须为正。')
        if not 1 <= self.history_steps <= MAX_HISTORY:
            raise RuntimeError(f'NPU history 必须在 1..{MAX_HISTORY}。')
        self.quiet = bool(quiet)
        self.perf_mode = str(perf_mode)
        self.rpc_latency = int(rpc_latency)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sessions = {}
        self.synced_at = 0.0
        self.sync_count = 0
        self.selected_ep_devices = []
        self.ep_device_summary = []
        self.stats = {
            'decisions': 0,
            'runs': 0,
            'real_actions': 0,
            'padded_actions': 0,
            'history_cpu_fallbacks': 0,
        }
        self._register_provider()
        self._make_run_options()
        self._allocate_buffers()
        self._probe_htp()
        self.sync(policy)

    def _make_run_options(self):
        # QNN documents these as per-run options.  Keeping performance mode here
        # avoids repeatedly injecting the same provider config keys during session
        # creation and is ideal for the many short calls made by a card-game actor.
        self.run_options = self.ort.RunOptions()
        self.run_options.add_run_config_entry('qnn.perf_mode', self.perf_mode)
        if self.rpc_latency >= 0:
            self.run_options.add_run_config_entry('qnn.rpc_control_latency', str(self.rpc_latency))

    def _allocate_buffers(self):
        h, a = self.history_steps, self.action_batch
        self.state_buf = np.zeros((1, STATE_DIM), np.float32)
        self.history_buf = np.zeros((1, h, EVENT_DIM), np.float32)
        self.valid_mask_buf = np.zeros((1, h), np.float32)
        self.last_mask_buf = np.zeros((1, h), np.float32)
        self.inv_length_buf = np.ones((1,), np.float32)
        self.actions_buf = np.zeros((a, ACTION_DIM), np.float32)
        self.feed = {
            'state': self.state_buf,
            'history': self.history_buf,
            'valid_mask': self.valid_mask_buf,
            'last_mask': self.last_mask_buf,
            'inv_length': self.inv_length_buf,
            'actions': self.actions_buf,
        }

    def _register_provider(self):
        """Register QNN plugin and select its real EP device object."""
        ort, qnn_ep = self.ort, self.qnn_ep
        register = getattr(ort, 'register_execution_provider_library', None)
        get_devices = getattr(ort, 'get_ep_devices', None)
        if register is None or get_devices is None:
            raise RuntimeError(
                '当前 onnxruntime 不具备插件 EP 设备 API；final-v3-f5 需要 Windows ARM64 上的 '
                'onnxruntime==1.27.0 与 onnxruntime-qnn==2.6.0。'
            )
        try:
            register('QNNExecutionProvider', qnn_ep.get_library_path())
        except Exception as e:
            try:
                already = [d for d in get_devices() if getattr(d, 'ep_name', None) == 'QNNExecutionProvider']
            except Exception:
                already = []
            if not already:
                raise RuntimeError(f'注册 QNN 插件失败：{e}') from e

        try:
            all_devices = list(get_devices())
        except Exception as e:
            raise RuntimeError(f'读取 ONNX Runtime EP 设备失败：{e}') from e
        selected = [d for d in all_devices if getattr(d, 'ep_name', None) == 'QNNExecutionProvider']
        if not selected:
            summary = [repr(d) for d in all_devices]
            raise RuntimeError('QNN 插件已注册，但 get_ep_devices() 没有返回 QNN 设备。可见设备：'+str(summary))
        self.selected_ep_devices = selected
        self.ep_device_summary = [repr(d) for d in selected]
        if getattr(self.ort.SessionOptions(), 'add_provider_for_devices', None) is None:
            raise RuntimeError(
                '当前 onnxruntime 缺少 SessionOptions.add_provider_for_devices()；请安装 README 指定的精确版本。'
            )

    def _provider_options(self):
        # Deliberately small option set. FP16-on-HTP is already the QNN default for
        # this runtime generation. burst/rpc latency are set per-run above. Fewer
        # duplicated config entries also eliminates the wall of harmless warnings.
        options = {'htp_graph_finalization_optimization_mode': '3'}
        get_htp = getattr(self.qnn_ep, 'get_qnn_htp_path', None)
        if get_htp is not None:
            options['backend_path'] = get_htp()
        else:
            options['backend_type'] = 'htp'
        return options

    def _attach_qnn_device(self, options):
        try:
            options.add_provider_for_devices(self.selected_ep_devices, self._provider_options())
        except Exception as e:
            raise RuntimeError(
                '把 QNN/HTP 设备绑定到 SessionOptions 失败：'+str(e)
                + f'；ORT={getattr(self.ort, "__version__", "?")} '
                + f'QNN={getattr(self.qnn_ep, "__version__", "?")} '
                + f'设备={self.ep_device_summary}'
            ) from e

    def _session_options(self, *, strict=True, profiling=False, profile_prefix=None):
        options = self.ort.SessionOptions()
        options.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        # Hide harmless QNN warning spam during periodic 4-model rebuilds. Fatal
        # errors still surface normally. --npu-verbose lowers this back to WARNING.
        if self.quiet:
            options.log_severity_level = 3  # ERROR
        if strict:
            options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
        if profiling:
            options.enable_profiling = True
            if profile_prefix is not None:
                options.profile_file_prefix = str(profile_prefix)
        self._attach_qnn_device(options)
        return options

    def _probe_htp(self):
        """Strict tiny Gemm probe before exporting the actor."""
        onnx = self.onnx
        from onnx import TensorProto, helper, numpy_helper
        path = self.cache_dir/'_qnn_probe.onnx'
        x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 4])
        y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 4])
        w = numpy_helper.from_array(np.eye(4, dtype=np.float32), name='w')
        b = numpy_helper.from_array(np.zeros(4, dtype=np.float32), name='b')
        node = helper.make_node('Gemm', ['x', 'w', 'b'], ['y'])
        graph = helper.make_graph([node], 'qnn_htp_probe', [x], [y], [w, b])
        model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 18)])
        model.ir_version = min(model.ir_version, 10)
        onnx.save(model, str(path))
        try:
            sess = self.ort.InferenceSession(str(path), sess_options=self._session_options(strict=True))
            out = sess.run(None, {'x': np.ones((1,4), np.float32)}, self.run_options)[0]
            if out.shape != (1,4) or not np.all(np.isfinite(out)):
                raise RuntimeError('HTP probe 返回异常张量。')
        except Exception as e:
            raise RuntimeError(
                'QNN/HTP 最小 Gemm 自检失败；问题发生在 QNN 设备绑定/运行时栈，尚未进入斗地主模型。'
                f' ORT={getattr(self.ort, "__version__", "?")} QNN={getattr(self.qnn_ep, "__version__", "?")} '
                f'设备={self.ep_device_summary}；原始错误：{e}'
            ) from e

    def _examples(self):
        return (
            torch.zeros((1, STATE_DIM), dtype=torch.float32),
            torch.zeros((1, self.history_steps, EVENT_DIM), dtype=torch.float32),
            torch.ones((1, self.history_steps), dtype=torch.float32),
            torch.zeros((1, self.history_steps), dtype=torch.float32),
            torch.ones((1,), dtype=torch.float32),
            torch.zeros((self.action_batch, ACTION_DIM), dtype=torch.float32),
        )

    def _export(self, name, net):
        path = self.cache_dir/f'{name}.onnx'
        tmp = self.cache_dir/f'.{name}.writing.onnx'
        if tmp.exists(): tmp.unlink()
        actor = _FixedQnnActor(net).eval()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='You are using the legacy TorchScript-based ONNX export.*',
                category=DeprecationWarning,
            )
            warnings.filterwarnings('ignore', message='ONNX export squeeze with negative axis.*', category=UserWarning)
            warnings.filterwarnings('ignore', message='This model contains a squeeze operation.*', category=UserWarning)
            torch.onnx.export(
                actor, self._examples(), tmp,
                input_names=('state', 'history', 'valid_mask', 'last_mask', 'inv_length', 'actions'),
                output_names=('values',), opset_version=18,
                do_constant_folding=True, dynamo=False,
            )
        os.replace(tmp, path)
        self._audit_onnx(path)
        return path

    def _audit_onnx(self, path):
        model = self.onnx.load(str(path), load_external_data=False)
        self.onnx.checker.check_model(model)
        from collections import Counter
        counts = Counter(node.op_type for node in model.graph.node)
        blocked_ops = {'Erf', 'ConstantOfShape', 'MatMul'}
        blocked = sorted(op for op in counts if op in blocked_ops)
        dynamic_inputs = []
        for value in model.graph.input:
            tensor_type = value.type.tensor_type
            if not tensor_type.HasField('shape'):
                dynamic_inputs.append(value.name); continue
            for dim in tensor_type.shape.dim:
                if dim.dim_param or not dim.HasField('dim_value') or dim.dim_value <= 0:
                    dynamic_inputs.append(value.name); break
        diag_dir = self.cache_dir/'diagnostics'; diag_dir.mkdir(parents=True, exist_ok=True)
        inventory = {
            'model': path.name,
            'operators': dict(sorted(counts.items())),
            'blocked_known_bad': blocked,
            'dynamic_inputs': dynamic_inputs,
            'npu_history_steps': self.history_steps,
            'npu_action_batch': self.action_batch,
        }
        (diag_dir/f'{path.stem}_graph.json').write_text(
            json.dumps(inventory, ensure_ascii=False, indent=2), encoding='utf-8')
        problems = []
        if blocked: problems.append('已知不适合当前浮点 HTP actor 的算子：'+str(blocked))
        if dynamic_inputs: problems.append('存在动态输入维度：'+str(dynamic_inputs))
        if problems:
            raise RuntimeError('；'.join(problems)+'。为避免静默 CPU fallback，已拒绝加载。')

    def _smoke_feed(self):
        state = np.zeros((1, STATE_DIM), np.float32)
        history = np.zeros((1, self.history_steps, EVENT_DIM), np.float32)
        valid = np.zeros((1, self.history_steps), np.float32); valid[0, 0] = 1.0
        last = np.zeros((1, self.history_steps), np.float32); last[0, 0] = 1.0
        return {
            'state': state,
            'history': history,
            'valid_mask': valid,
            'last_mask': last,
            'inv_length': np.asarray([1.0], np.float32),
            'actions': np.zeros((self.action_batch, ACTION_DIM), np.float32),
        }

    def _diagnose_cpu_fallback(self, path):
        """Run once with CPU fallback enabled and report nodes that land on CPU."""
        diag_dir = self.cache_dir/'diagnostics'; diag_dir.mkdir(parents=True, exist_ok=True)
        options = self._session_options(
            strict=False, profiling=True, profile_prefix=diag_dir/f'{path.stem}_ort')
        try:
            session = self.ort.InferenceSession(str(path), sess_options=options)
            session.run(['values'], self._smoke_feed(), self.run_options)
            profile = Path(session.end_profiling())
            data = json.loads(profile.read_text(encoding='utf-8'))
            cpu_nodes = []
            for event in data:
                args = event.get('args') or {}
                if args.get('provider') != 'CPUExecutionProvider': continue
                label = args.get('op_name') or event.get('name') or 'unknown'
                node = args.get('node_index')
                text = f'{label}#{node}' if node is not None else str(label)
                if text not in cpu_nodes: cpu_nodes.append(text)
            if cpu_nodes:
                preview = ', '.join(cpu_nodes[:12])
                more = f'（另有 {len(cpu_nodes)-12} 个）' if len(cpu_nodes) > 12 else ''
                return f'CPU fallback 节点：{preview}{more}；ORT profile：{profile}'
            return f'诊断 session 可运行，但 profile 未标出 CPU 节点；请查看：{profile}'
        except Exception as diagnostic_error:
            return f'CPU-fallback 诊断也失败：{diagnostic_error}'

    def _session(self, path):
        try:
            return self.ort.InferenceSession(str(path), sess_options=self._session_options(strict=True))
        except Exception as e:
            diagnostic = self._diagnose_cpu_fallback(path)
            raise RuntimeError(f'Hexagon NPU 无法完整接管模型 {path.name}：{e}；{diagnostic}') from e

    def sync(self, policy):
        was_training = policy.training
        policy.eval()
        try:
            sessions = {}
            for name in BUCKETS:
                path = self._export(name, policy.part(name))
                sessions[name] = self._session(path)
            # Warm every role once. This shifts first-run setup latency into the
            # periodic sync and keeps game timing smooth afterwards.
            smoke = self._smoke_feed()
            for name, session in sessions.items():
                out = session.run(['values'], smoke, self.run_options)[0]
                if out.shape[0] != self.action_batch or not np.all(np.isfinite(out)):
                    raise RuntimeError(f'NPU {name} smoke test 返回异常结果。')
            self.sessions = sessions
            self.synced_at = time.monotonic()
            self.sync_count += 1
        finally:
            policy.train(was_training)

    def values(self, state_public, actions, return_features=False):
        state, history, candidates = prepare(state_public, actions)
        if len(history) > self.history_steps:
            self.stats['history_cpu_fallbacks'] += 1
            raise NpuStepFallback(
                f'当前历史 {len(history)} > NPU 快速上限 {self.history_steps}；本步改用 CPU。')
        part = model_part_for(state_public)
        n_hist = max(1, len(history))

        # Reuse fixed buffers instead of allocating six NumPy arrays every decision.
        self.state_buf[0, :] = state
        self.history_buf.fill(0.0)
        self.history_buf[0, :len(history), :] = history
        self.valid_mask_buf.fill(0.0)
        self.valid_mask_buf[0, :len(history)] = 1.0
        self.last_mask_buf.fill(0.0)
        if len(history): self.last_mask_buf[0, len(history)-1] = 1.0
        self.inv_length_buf[0] = 1.0/n_hist

        values = []
        real = len(candidates)
        runs = 0
        for start in range(0, real, self.action_batch):
            chunk = candidates[start:start+self.action_batch]
            self.actions_buf.fill(0.0)
            self.actions_buf[:len(chunk), :] = chunk
            result = self.sessions[part].run(['values'], self.feed, self.run_options)[0]
            values.append(np.asarray(result[:len(chunk)], dtype=np.float32))
            runs += 1
        self.stats['decisions'] += 1
        self.stats['runs'] += runs
        self.stats['real_actions'] += real
        self.stats['padded_actions'] += runs*self.action_batch
        q = np.concatenate(values)
        return (q, (state, history, candidates)) if return_features else q

    def perf_summary(self):
        d = max(1, self.stats['decisions'])
        real = max(1, self.stats['real_actions'])
        return (
            f'NPU {self.stats["runs"]/d:.2f} run/决策, '
            f'动作槽×{self.stats["padded_actions"]/real:.2f}, '
            f'长历史CPU回退 {self.stats["history_cpu_fallbacks"]}'
        )


class RolloutBackend:
    def __init__(self, args, agent, policy, policy_lock=None):
        self.cpu = agent
        self.policy_lock = policy_lock
        self.npu = None
        self.name = 'CPU(PyTorch)'
        self.last_error = None
        want_npu = args.backend == 'npu' or (
            args.backend == 'auto' and platform.system() == 'Windows'
            and platform.machine().lower() in ('arm64', 'aarch64'))
        if want_npu:
            try:
                self.npu = QnnRollout(
                    policy, args.npu_cache,
                    action_batch=args.npu_action_batch,
                    history_steps=args.npu_history,
                    quiet=not args.npu_verbose,
                    perf_mode=args.npu_perf_mode,
                    rpc_latency=args.npu_rpc_latency,
                )
                self.name = 'Hexagon NPU(QNN rollout) + CPU learner'
                print(
                    f'已启用 Hexagon NPU：固定历史 {args.npu_history}，动作批 {args.npu_action_batch}；'
                    '自对弈前向走 QNN，反向传播由 CPU/PyTorch 完成。', flush=True)
            except Exception as e:
                self.last_error = str(e)
                if args.backend == 'npu':
                    raise RuntimeError('强制 NPU 模式初始化失败：'+str(e)) from e
                print('NPU 自动启用失败，已安全回退 CPU：'+str(e), flush=True)
        elif args.backend == 'npu':
            raise RuntimeError('当前系统不是 Windows ARM64，不能强制使用 Snapdragon Hexagon NPU。')

    def _cpu_values(self, s, actions, return_features):
        if self.policy_lock is None:
            return self.cpu.values(s, actions, return_features)
        with self.policy_lock:
            return self.cpu.values(s, actions, return_features)

    def values(self, s, actions, return_features=False):
        if self.npu is not None:
            try:
                return self.npu.values(s, actions, return_features)
            except NpuStepFallback:
                # Keep the NPU alive; only this rare oversized-history state uses CPU.
                return self._cpu_values(s, actions, return_features)
            except Exception as e:
                self.last_error = str(e)
                print('NPU rollout 发生错误，本次起回退 CPU：'+str(e), file=sys.stderr, flush=True)
                self.npu = None
                self.name = 'CPU(PyTorch，NPU 已回退)'
        return self._cpu_values(s, actions, return_features)

    def maybe_sync(self, policy, interval):
        if self.npu is None:
            return False
        if time.monotonic()-self.npu.synced_at < interval:
            return False
        try:
            self.npu.sync(policy)
            return True
        except Exception as e:
            self.last_error = str(e)
            print('NPU 权重同步失败，后续回退 CPU：'+str(e), file=sys.stderr, flush=True)
            self.npu = None
            self.name = 'CPU(PyTorch，NPU 同步失败后回退)'
            return False

    def sync_due(self, interval):
        return self.npu is not None and time.monotonic()-self.npu.synced_at >= interval

    def perf_summary(self):
        return self.npu.perf_summary() if self.npu is not None else ''

def train(args, stop):
    # On an NPU host, leave two logical CPUs free for Python game logic, ORT/QNN
    # dispatch and the OS.  The PyTorch learner then uses the remaining cores.
    auto_npu_host = (
        args.backend != 'cpu' and platform.system() == 'Windows'
        and platform.machine().lower() in ('arm64', 'aarch64')
    )
    if args.threads > 0:
        threads = args.threads
    else:
        # f6 runs four independent role learners concurrently.  Giving every small
        # 1M network all CPU cores causes severe intra-op oversubscription; two
        # intra-op workers per update is a much better default on 12-core X Elite.
        threads = 1 if auto_npu_host else max(1, min(2, (os.cpu_count() or 4)//max(1, len(BUCKETS))))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    rng = random.Random(args.seed)
    learner_rng = {name: random.Random(args.seed ^ 0x5A17D0C3 ^ (i*0x9E3779B1)) for i, name in enumerate(BUCKETS)}
    torch.manual_seed(args.seed)

    ensure_model(args.model, args.checkpoint, args.seed)
    agent = Agent(args.model, threads)
    policy = agent.policy
    if agent.meta.get('rules') != DEFAULT_RULES:
        raise ValueError('模型的规则不是本包默认规则，不能继续训练。')

    optimizers = {
        name: torch.optim.AdamW(policy.part(name).parameters(), lr=args.lr, weight_decay=1e-5, foreach=True)
        for name in BUCKETS
    }
    replay = {name: deque(maxlen=args.buffer) for name in BUCKETS}
    recent_losses = {name: deque(maxlen=100) for name in BUCKETS}
    recent_role_wins = {name: deque(maxlen=200) for name in ROLES}
    frames = games = updates = 0
    updates_by_part = {name: 0 for name in BUCKETS}
    previous_seconds = 0.0

    if args.checkpoint.is_file() and not args.fresh:
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        check_bundle(saved)
        saved_schema = int(saved.get('training_schema', 0))
        if saved_schema not in (2, 3, 4, TRAINING_SCHEMA) or saved['meta'].get('rules') != DEFAULT_RULES:
            raise ValueError('续训文件不是兼容的 final-v3 格式；请恢复配套文件，或改名备份后用 --fresh。')
        if saved['meta'].get('run_id') != agent.meta.get('run_id'):
            raise ValueError('current.pt 和 training.pt 来自不同训练。请恢复配套文件，或用 --fresh。')
        if saved['meta'].get('frames', 0) < agent.meta.get('frames', 0):
            raise ValueError('training.pt 比 current.pt 更旧；请恢复配套文件，或用 --fresh。')
        policy.load_state_dict(saved['weights'], strict=True)
        for name in BUCKETS:
            optimizers[name].load_state_dict(saved['optimizers'][name])
            for group in optimizers[name].param_groups:
                group['lr'] = args.lr
        agent.meta = saved['meta']
        frames, games, updates = (int(saved['meta'].get(k, 0)) for k in ('frames', 'games', 'updates'))
        updates_by_part.update({k: int(v) for k, v in saved['meta'].get('updates_by_part', {}).items() if k in BUCKETS})
        previous_seconds = float(saved['meta'].get('training_seconds', 0.0))
        rng.setstate(saved['rng'])
        if 'learner_rng' in saved:
            old_rng = saved['learner_rng']
            if isinstance(old_rng, dict):
                for name in BUCKETS:
                    if name in old_rng: learner_rng[name].setstate(old_rng[name])
            else:
                # schema <=4 stored one sampler RNG; deterministic continuation is
                # impossible after switching to four workers, but weights/replay/optimizer are exact.
                for i, name in enumerate(BUCKETS): learner_rng[name].seed(args.seed ^ 0x5A17D0C3 ^ (i*0x9E3779B1) ^ updates)
        torch.set_rng_state(saved['torch_rng'])
        for name in BUCKETS:
            replay[name].extend(saved['replay'][name][-args.buffer:])
            recent_losses[name].extend(saved.get('recent_losses', {}).get(name, []))
        if saved_schema >= 3:
            for name in ROLES:
                recent_role_wins[name].extend(saved.get('recent_role_wins', {}).get(name, []))
        elif saved.get('recent_landlord_wins'):
            print('旧续训文件没有上下家独立终结统计；三席近期胜率窗口从本次继续训练起重新累计。', flush=True)
        print(f'已恢复：{games} 局 / {frames} 决策 / {updates} 更新；回放 {sum(map(len, replay.values()))} 条。', flush=True)
        del saved
    elif args.fresh:
        frames = int(agent.meta.get('frames', 0)); games = int(agent.meta.get('games', 0))
        updates = int(agent.meta.get('updates', 0)); previous_seconds = float(agent.meta.get('training_seconds', 0.0))
        updates_by_part.update(agent.meta.get('updates_by_part', {}))
        print('从 current.pt 已学权重继续；重新建立四个优化器和回放。', flush=True)
    else:
        frames = int(agent.meta.get('frames', 0)); games = int(agent.meta.get('games', 0))
        updates = int(agent.meta.get('updates', 0)); previous_seconds = float(agent.meta.get('training_seconds', 0.0))
        updates_by_part.update(agent.meta.get('updates_by_part', {}))
        print('没有续训状态；从 current.pt 建立新的优化器与回放。', flush=True)

    policy.eval()
    part_locks = {name: threading.RLock() for name in BUCKETS}
    replay_locks = {name: threading.RLock() for name in BUCKETS}

    class _AllPartLocks:
        def __enter__(self):
            for name in BUCKETS: part_locks[name].acquire()
            return self
        def __exit__(self, *exc):
            for name in reversed(BUCKETS): part_locks[name].release()
    all_part_locks = _AllPartLocks()

    # CPU fallback is rare in NPU mode.  It must see a coherent role network, but
    # need not block unrelated role learners.
    class _CpuPolicyLock:
        def __enter__(self):
            all_part_locks.__enter__(); return self
        def __exit__(self, *exc):
            return all_part_locks.__exit__(*exc)
    rollout = RolloutBackend(args, agent, policy, policy_lock=_CpuPolicyLock())
    async_learner = bool(args.async_learner and rollout.npu is not None)

    start = last_save = last_report = time.monotonic()
    initial_frames, initial_games, initial_updates = frames, games, updates
    learner_error = []
    learner_shutdown = threading.Event()
    budget_cv = threading.Condition()
    pending_by_part = {name: 0 for name in BUCKETS}
    learner_threads = []

    def epsilon():
        progress = min(1.0, frames/max(args.epsilon_steps, 1))
        return args.epsilon+(args.epsilon_end-args.epsilon)*progress

    def loss_mean(name=None):
        if name is not None:
            return float(np.mean(recent_losses[name])) if recent_losses[name] else None
        values = [v for q in recent_losses.values() for v in q]
        return float(np.mean(values)) if values else None

    def meta():
        elapsed = time.monotonic()-start
        counts = policy.parameter_counts()
        return {
            **agent.meta,
            'source': 'QOJ final-v3-f6 four independent role Q networks; parallel CPU learner; Monte Carlo self-play',
            'rules': dict(DEFAULT_RULES), 'frames': frames, 'games': games, 'updates': updates,
            'updates_by_part': dict(updates_by_part),
            'training_seconds': previous_seconds+elapsed,
            'parameters': sum(counts.values()), 'parameters_by_part': counts,
            'epsilon': epsilon(), 'loss': loss_mean(),
            'loss_by_part': {k: loss_mean(k) for k in BUCKETS},
            'rollout_backend': rollout.name,
            'async_learner': async_learner,
            'npu_action_batch': args.npu_action_batch,
            'npu_history': args.npu_history,
            'saved_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        }

    def report(saved=False):
        elapsed = time.monotonic()-start
        rate = (frames-initial_frames)/max(elapsed, 1e-9)
        loss = loss_mean()
        loss_text = f'{loss:.4f}' if loss is not None else '等待样本'
        if any(recent_role_wins[name] for name in ROLES):
            def pct(name):
                q = recent_role_wins[name]
                return 100.0*float(np.mean(q)) if q else 0.0
            wins = (f'地主 {pct("landlord"):.1f}% / '
                    f'地主下家 {pct("landlord_down"):.1f}% / '
                    f'地主上家 {pct("landlord_up"):.1f}%')
        else:
            wins = '地主 — / 地主下家 — / 地主上家 —'
        with budget_cv:
            pending = sum(pending_by_part.values())
            pending_parts = '/'.join(str(pending_by_part[k]) for k in BUCKETS)
        extra = rollout.perf_summary()
        if async_learner:
            update_rate = (updates-initial_updates)/max(elapsed, 1e-9)
            budget_rate = (games-initial_games)*args.updates_per_game/max(elapsed, 1e-9)
            extra = (extra + ' | ' if extra else '') + (
                f'learner {update_rate:.1f} 更新/s, 预算 {budget_rate:.1f}/s, '
                f'待更新 {pending} [{pending_parts}]')
        print(
            f'{"已保存" if saved else "训练中"}：{games} 局 / {frames} 决策 / {updates} 更新'
            f' | {rate:.1f} 决策/s | loss {loss_text} | ε {epsilon():.3f}'
            f' | 近200局三席独立胜率 {wins} | {rollout.name}'
            + (f' | {extra}' if extra else ''), flush=True)

    def should_stop():
        if learner_error:
            return True
        return stop.is_set() or (args.seconds > 0 and time.monotonic()-start >= args.seconds)

    def do_one_update(name):
        """One optimizer step for exactly one independent role network."""
        nonlocal updates
        with replay_locks[name]:
            n = len(replay[name])
            if n < args.min_samples:
                return False
            # deque indexing is O(n); materialize once per update.  With a 4096
            # replay this is cheap, while keeping checkpoint schema compatible.
            pool = list(replay[name])
            rr = learner_rng[name]
            # Length-bucketed *uniform-marginal* sampling. Pick an anchor uniformly,
            # then sample from its 16-step history bucket. A bucket is therefore
            # chosen proportional to its population, so every replay row keeps the
            # same marginal probability while pad_sequence wastes far less compute.
            anchor = pool[rr.randrange(n)]
            bucket_id = len(anchor['history']) // 16
            local = [r for r in pool if len(r['history']) // 16 == bucket_id]
            rows = (rr.sample(local, args.batch) if len(local) >= args.batch
                    else rr.choices(local, k=args.batch))
        states, histories, lengths, actions, targets = make_batch(rows)
        with part_locks[name]:
            net = policy.part(name)
            net.train()
            predicted = policy(name, states, histories, lengths, actions)
            loss = torch.nn.functional.smooth_l1_loss(predicted, targets, beta=args.huber_beta)
            if not torch.isfinite(loss):
                raise RuntimeError(f'{name} 训练出现非有限损失；请恢复之前保存的模型。')
            opt = optimizers[name]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net.parameters(), args.grad_clip, error_if_nonfinite=True, foreach=True)
            opt.step()
            net.eval()
            updates_by_part[name] += 1
            recent_losses[name].append(float(loss.detach()))
        # Python integer update is protected because += is not a synchronization primitive.
        with budget_cv:
            updates += 1
        return True

    def learn_sync(count):
        # Preserve f5 round-robin semantics in CPU-only mode.
        for i in range(count):
            if should_stop(): break
            name = BUCKETS[i % len(BUCKETS)]
            if not do_one_update(name):
                # fall back to any warm role
                for alt in BUCKETS:
                    if do_one_update(alt): break

    def queue_async_updates(count):
        # Bound learner debt.  Once the optimized learner reaches the configured
        # small backlog, throttle rollout instead of accumulating unbounded fake
        # throughput. This preserves exactly updates_per_game optimizer steps.
        if args.max_pending > 0:
            with budget_cv:
                budget_cv.wait_for(
                    lambda: sum(pending_by_part.values()) < args.max_pending
                    or learner_shutdown.is_set() or stop.is_set(), timeout=1.0)
                while (sum(pending_by_part.values()) >= args.max_pending
                       and not learner_shutdown.is_set() and not stop.is_set()):
                    budget_cv.wait(timeout=0.05)
        # f5 uses four updates/game and round-robins four roles once replay is warm.
        # Generalize counts without changing the total update budget.
        warm = []
        for name in BUCKETS:
            with replay_locks[name]:
                if len(replay[name]) >= args.min_samples: warm.append(name)
        if not warm: return
        with budget_cv:
            for i in range(count):
                pending_by_part[warm[i % len(warm)]] += 1
            budget_cv.notify_all()

    def learner_loop(name):
        try:
            while not learner_shutdown.is_set():
                with budget_cv:
                    budget_cv.wait_for(
                        lambda: pending_by_part[name] > 0 or learner_shutdown.is_set() or stop.is_set(),
                        timeout=0.25)
                    if learner_shutdown.is_set() or stop.is_set(): break
                    if pending_by_part[name] <= 0: continue
                ok = do_one_update(name)
                with budget_cv:
                    if ok and pending_by_part[name] > 0:
                        pending_by_part[name] -= 1
                    elif not ok:
                        pending_by_part[name] = 0
                    budget_cv.notify_all()
        except BaseException as e:
            learner_error.append((name, e))
            stop.set()
            with budget_cv: budget_cv.notify_all()

    def snapshot_for_save():
        # Model weights and optimizer moments must come from the SAME learner step.
        # Take them in one policy-lock critical section, then perform slow disk I/O
        # after releasing the learner.
        with all_part_locks:
            model = copy.deepcopy(bundle(policy, meta()))
            opt_states = copy.deepcopy({k: v.state_dict() for k, v in optimizers.items()})
            torch_state = torch.get_rng_state().clone()
        replay_state = {}
        for k in BUCKETS:
            with replay_locks[k]: replay_state[k] = list(replay[k])
        learner_rng_state = {k: learner_rng[k].getstate() for k in BUCKETS}
        checkpoint = {
            **model, 'training_schema': TRAINING_SCHEMA,
            'optimizers': opt_states,
            'rng': rng.getstate(), 'learner_rng': learner_rng_state, 'torch_rng': torch_state,
            'replay': replay_state,
            'recent_losses': {k: list(v) for k, v in recent_losses.items()},
            'recent_role_wins': {k: list(v) for k, v in recent_role_wins.items()},
        }
        return model, checkpoint

    def save(sync_npu=True):
        nonlocal last_save, last_report
        model, checkpoint = snapshot_for_save()
        atomic_save(checkpoint, args.checkpoint)
        atomic_save(model, args.model)
        agent.meta = model['meta']
        last_save = last_report = time.monotonic()
        report(True)
        if sync_npu and rollout.sync_due(args.npu_sync_interval):
            # Compile a coherent weight snapshot while learner continues updating the
            # live PyTorch policy in the background.
            with all_part_locks:
                actor_snapshot = copy.deepcopy(policy).eval()
            if rollout.maybe_sync(actor_snapshot, args.npu_sync_interval):
                print('NPU actor 已同步最新四子模型权重。', flush=True)
            del actor_snapshot

    def maintenance():
        nonlocal last_report
        if learner_error:
            raise RuntimeError(f'后台 learner 失败：{learner_error[0][0]}: {learner_error[0][1]}')
        now = time.monotonic()
        if now-last_save >= args.save_interval:
            save()
        elif now-last_report >= args.log_interval:
            report(); last_report = now

    print(f'CPU learner：4 路并行，每路 PyTorch intra-op {threads} 线程；模型总参数 {sum(p.numel() for p in policy.parameters()):,}。', flush=True)
    print('四部分：' + ' / '.join(f'{k} {v:,}' for k, v in policy.parameter_counts().items()), flush=True)
    if rollout.npu is not None:
        print(
            f'NPU 优化：历史固定 {args.npu_history}/{MAX_HISTORY}，动作批 {args.npu_action_batch}；'
            f'QNN 日志{"保留" if args.npu_verbose else "静默警告"}；'
            f'异步 CPU learner {"开启" if async_learner else "关闭"}。', flush=True)
    save(sync_npu=False)

    if async_learner:
        for name in BUCKETS:
            t = threading.Thread(target=learner_loop, args=(name,), name=f'qoj-learner-{name}', daemon=True)
            t.start(); learner_threads.append(t)

    try:
        while not should_stop() and (not args.games or games-initial_games < args.games):
            game = Game(rng.getrandbits(63)); trajectory = []
            for _ in range(256):
                if should_stop():
                    break
                state = game.observation(); actions = legal_actions(state)
                q, (x, history, candidates) = rollout.values(state, actions, return_features=True)
                index = rng.randrange(len(actions)) if rng.random() < epsilon() else int(q.argmax())
                part = model_part_for(state)
                row = {
                    'state': torch.from_numpy(x.copy()),
                    'history': torch.from_numpy(history.copy()),
                    'action': torch.from_numpy(candidates[index].copy()),
                }
                trajectory.append((part, state['seat'], row))
                game.apply(actions[index]); frames += 1
                if game.phase == 'finished':
                    for name, seat, sample in trajectory:
                        sample['target'] = float(game.result['deltas'][seat])/SCORE_SCALE
                        with replay_locks[name]: replay[name].append(sample)
                    games += 1
                    winner_role = role_for(game.result['winner_seat'], game.landlord)
                    for role_name in ROLES:
                        recent_role_wins[role_name].append(int(role_name == winner_role))
                    if async_learner:
                        queue_async_updates(args.updates_per_game)
                    else:
                        learn_sync(args.updates_per_game)
                    break
                maintenance()
            else:
                raise RuntimeError('单局超过 256 次动作；已停止，请检查规则。')
            maintenance()

        # For a normal --games/--seconds completion, let already-scheduled work catch
        # up so updates-per-game remains meaningful. Ctrl+C intentionally stops fast.
        if async_learner and not stop.is_set() and not learner_error:
            while True:
                with budget_cv:
                    pending = sum(pending_by_part.values())
                if pending <= 0:
                    break
                time.sleep(0.01)
                maintenance()

        learner_shutdown.set()
        with budget_cv:
            budget_cv.notify_all()
        for t in learner_threads:
            t.join()
        if learner_error:
            raise RuntimeError(f'后台 learner 失败：{learner_error[0][0]}: {learner_error[0][1]}')
        save()
    except Exception:
        learner_shutdown.set()
        with budget_cv:
            budget_cv.notify_all()
        for t in learner_threads:
            if t.is_alive(): t.join(timeout=5)
        print('训练异常退出；保留上一次成功保存的模型和续训文件。', file=sys.stderr, flush=True)
        raise

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=('auto', 'cpu', 'npu'), default='auto',
                   help='auto：Windows ARM64 尝试 Hexagon NPU rollout，否则 CPU；npu 为强制模式')
    p.add_argument('--threads', type=int, default=0, help='每路 learner 的 PyTorch intra-op 线程；0=自动（Surface X Elite 默认 2；四角色并行）')
    p.add_argument('--batch', type=int, default=64, help='每个子模型一次更新的样本数')
    p.add_argument('--buffer', type=int, default=4096, help='四个角色各自的 replay 容量')
    p.add_argument('--min-samples', type=int, default=64, help='某角色积累到此样本数后开始更新')
    p.add_argument('--updates-per-game', type=int, default=4, help='每局完成后的梯度更新次数')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--huber-beta', type=float, default=1.0, help='SmoothL1/Huber 转折点')
    p.add_argument('--grad-clip', type=float, default=5.0)
    p.add_argument('--epsilon', type=float, default=0.20, help='初始随机探索率')
    p.add_argument('--epsilon-end', type=float, default=0.05)
    p.add_argument('--epsilon-steps', type=int, default=2_000_000, help='探索率线性退火决策数')
    p.add_argument('--save-interval', type=float, default=60, help='保存间隔；与 v2 默认频率相同')
    p.add_argument('--log-interval', type=float, default=10, help='汇报间隔；与 v2 默认频率相同')
    p.add_argument('--npu-sync-interval', type=float, default=60,
                   help='NPU actor 重新导出/加载最新权重的最短秒数')
    p.add_argument('--npu-action-batch', type=int, default=NPU_ACTION_BATCH_DEFAULT,
                   choices=(8, 16, 32, 64, 128, 256),
                   help='NPU 每次固定动作槽；默认16，减少原256槽的大量 padding')
    p.add_argument('--npu-history', type=int, default=NPU_HISTORY_DEFAULT,
                   choices=(64, 96, 128, 160, 192, 256, 320),
                   help='NPU 固定历史长度；超出时仅该一步回 CPU，模型本身仍保留320')
    p.add_argument('--npu-perf-mode', default='burst',
                   choices=('burst', 'high_performance', 'sustained_high_performance', 'balanced'),
                   help='QNN HTP 每次推理的性能模式')
    p.add_argument('--npu-rpc-latency', type=int, default=100,
                   help='QNN RPC control latency（微秒）；-1=不设置，默认100')
    p.add_argument('--npu-verbose', action='store_true',
                   help='显示 ONNX Runtime/QNN WARNING；默认隐藏周期同步产生的无害警告')
    p.add_argument('--max-pending', type=int, default=256,
                   help='异步 learner 最大更新欠账；达到后暂停 rollout 等 learner 追上。0=不限制（不推荐）')
    p.add_argument('--async-learner', action=argparse.BooleanOptionalAction, default=True,
                   help='NPU 模式下后台并行 CPU learner；可用 --no-async-learner 关闭做对照')
    p.add_argument('--games', type=int, default=0, help='本次再完成多少局，0=不限')
    p.add_argument('--seconds', type=float, default=0, help='本次训练秒数上限，0=不限')
    p.add_argument('--seed', type=int, default=20261005)
    p.add_argument('--fresh', action='store_true', help='保留 current.pt 权重，但重建优化器与 replay')
    p.add_argument('--model', type=Path, default=ROOT/'client/models/current.pt')
    p.add_argument('--checkpoint', type=Path, default=ROOT/'training.pt')
    p.add_argument('--npu-cache', type=Path, default=ROOT/'.npu_cache', help='自动生成的四个 ONNX 临时 actor')
    args = p.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf8', errors='replace')
    if args.threads < 0:
        p.error('--threads 不能为负；0 表示自动调优。')
    if args.max_pending < 0:
        p.error('--max-pending 不能为负。')
    if min(args.batch, args.buffer, args.min_samples, args.updates_per_game, args.epsilon_steps) < 1:
        p.error('批量、容量、最少样本、更新次数和探索步数必须为正。')
    if args.min_samples > args.buffer:
        p.error('--min-samples 不能大于 --buffer。')
    if min(args.lr, args.huber_beta, args.grad_clip, args.save_interval, args.log_interval, args.npu_sync_interval) <= 0:
        p.error('学习率、Huber beta、梯度裁剪和各时间间隔必须为正。')
    if not all(0 <= x <= 1 for x in (args.epsilon, args.epsilon_end)):
        p.error('探索率必须在 0–1 之间。')
    if min(args.games, args.seconds) < 0:
        p.error('局数、秒数不能为负。')
    if args.npu_rpc_latency < -1:
        p.error('--npu-rpc-latency 只能为 -1 或非负微秒。')

    args.model = args.model.resolve(); args.checkpoint = args.checkpoint.resolve(); args.npu_cache = args.npu_cache.resolve()
    if args.model == args.checkpoint:
        p.error('模型路径和续训路径不能相同。')
    stop = threading.Event()

    def request_stop(signum, frame):
        if not stop.is_set():
            stop.set()
            print('\n收到停止请求；完成当前计算并保存，请等待“训练已停止”。', flush=True)

    previous = signal.signal(signal.SIGINT, request_stop)
    try:
        with TrainingLock(args.model):
            train(args, stop)
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as error:
        print(f'训练未完成：{error}', file=sys.stderr, flush=True)
        return 1
    finally:
        signal.signal(signal.SIGINT, previous)
    print('训练已停止，最新模型已保存。可运行 python local.py 或 python client/qoj_cli.py。', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
