# QOJ 斗地主 final-v3-f5

final-v3-f5 基于已经可以在 Surface Pro 11 / Snapdragon X Elite Hexagon NPU 上运行的 f4 继续做性能优化。模型本身仍是四个完全独立的 Q 网络，总参数 **3,976,804**；`current.pt` 权重形状没有变化，旧 final-v3/f1/f2/f3/f4 的 schema-3 模型可以继续加载，旧 `training.pt` 也支持迁移续训。

## 1. 这版解决的“每隔一段时间打印一大坨东西”是什么

f4 每次保存后会定期把最新四个子模型重新导出/加载到 QNN。ONNX Runtime/QNN 在创建四个新 session 时会重复打印类似：

```text
Config with key [ep.qnnexecutionprovider.htp_performance_mode] already exists...
Config with key [ep.qnnexecutionprovider.backend_path] already exists...
File mapped weights feature is only available...
```

这些是 **WARNING，不是训练数据，也不是模型错误**：

- `already exists ... overwritten`：同一个 QNN provider 配置在插件 EP 创建过程中被重复写入，最终值仍然正确。
- `File mapped weights ... disabled`：当前设备暴露的 QNN API 不满足该可选优化功能的版本条件，运行库自动关闭它；不影响正常 HTP/NPU 推理。
- 它们通常会和 `NPU actor 已同步最新四子模型权重。` 一起出现，因为同步会重建四个 QNN session。

f5 默认把 QNN/ORT 的普通 WARNING 静默，只保留真正的 ERROR，并且把 `burst` / RPC latency 改成 QNN 官方提供的 **per-run** 配置，减少 session 创建阶段的重复配置。训练日志会重新保持简洁。

如果排障时需要看完整 QNN warning：

```powershell
python train.py --npu-verbose
```

## 2. f5 的性能优化

### 2.1 NPU 动作固定槽：256 -> 16

f4 为了最先确保固定 shape 可以完整上 HTP，每次推理都固定输入：

```text
actions = [256, 57]
```

即使当前只有 2、5、10 个合法动作，也仍然给其余位置补零并让动作网络计算。

用当前规则随机模拟 500 局、31,707 个决策得到：

```text
平均合法动作       5.63
中位数             2
90%                <= 11
95%                <= 21
99%                <= 64
```

所以 f5 默认：

```text
--npu-action-batch 16
```

合法动作超过 16 时自动分块，多次 NPU 调用后再拼回完整 Q 值。数学结果不变。

同一批样本中：

```text
旧 256 槽：平均计算约 256.0 个动作槽 / 决策
新 16 槽： 平均计算约 18.2 个动作槽 / 决策
           平均约 1.14 次 NPU run / 决策
```

这会大幅减少 action encoder / value head 的无效 padding 计算。

可手动比较：

```powershell
python train.py --npu-action-batch 8
python train.py --npu-action-batch 16
python train.py --npu-action-batch 32
python train.py --npu-action-batch 64
```

默认推荐 16。

### 2.2 NPU 历史固定长度：320 -> 128

模型本身仍支持：

```text
MAX_HISTORY = 320
```

但 HTP 固定 shape actor 默认只编译：

```text
history = [1, 128, 68]
```

相同随机 500 局样本里：

```text
历史平均长度       34.3
90%                <= 62
95%                <= 68
99%                <= 78
最大               107
```

因此默认 128 可以显著减少 history MLP 的 padding 计算。

如果真实训练中偶尔出现历史 > 128：

```text
只把该一个决策交给 CPU Policy
NPU session 不会失效
下一步仍继续使用 NPU
```

所以没有截断历史，也不会改变模型看到的信息。

可改为更保守：

```powershell
python train.py --npu-history 160
python train.py --npu-history 192
python train.py --npu-history 320
```

### 2.3 复用 NPU 输入缓冲区

f4 每个决策都会重新分配：

```text
state
history padding
valid mask
last mask
inv length
action padding
```

f5 在 QnnRollout 初始化时一次创建固定 NumPy buffer，后续只覆盖内容，减少 Python/NumPy 分配、垃圾回收和 ORT 输入包装压力。

### 2.4 QNN per-run 性能选项

f5 按 QNN plugin EP 官方接口建立一个长期复用的 `RunOptions`：

```text
qnn.perf_mode = burst
qnn.rpc_control_latency = 100 us
```

默认参数：

```text
--npu-perf-mode burst
--npu-rpc-latency 100
```

长时间训练如果机器因为温度/功耗出现明显降频，可以对比：

```powershell
python train.py --npu-perf-mode sustained_high_performance
```

### 2.5 CPU learner 与 NPU rollout 并行

f4 的流程基本是：

```text
NPU 自对弈
   ↓ 停
CPU backward / AdamW
   ↓
NPU 自对弈
```

f5 在 NPU 成功启用后默认开启后台 learner：

```text
主线程：CPU 游戏规则/特征 -> NPU rollout -> 下一步 -> ...
                              ||
后台：                    PyTorch CPU backward + AdamW
```

每完成一局仍然产生和原来相同的：

```text
updates_per_game = 4
```

只是这些更新进入后台预算，由 CPU learner 消费。训练目标、Huber loss、四角色 replay 和 optimizer 都没改。

在正常 `--games` / `--seconds` 完成时，程序会把已经排队的更新做完后再最终保存；Ctrl+C 则优先快速安全停止。

关闭异步做 A/B 对照：

```powershell
python train.py --no-async-learner
```

### 2.6 NPU 主机自动给游戏线程留 CPU

如果 `--threads 0`（默认）：

- 普通 CPU/Linux：PyTorch learner 使用全部逻辑 CPU。
- Windows ARM64 + NPU 模式：默认保留 2 个逻辑 CPU 给 Python 游戏逻辑、特征编码、ORT/QNN 调度和系统，其余给 learner。

Surface 12 核通常会显示：

```text
CPU learner 线程：10
```

仍然可以手工覆盖：

```powershell
python train.py --threads 8
python train.py --threads 10
python train.py --threads 12
```

建议实际比较 `决策/s`，不要只看 CPU/NPU 利用率。

## 3. 新增性能统计

NPU 模式日志会额外给出类似：

```text
NPU 1.14 run/决策, 动作槽×3.23, 长历史CPU回退 0 | 异步learner 待更新 2
```

解释：

- `run/决策`：平均每个斗地主决策需要多少次 QNN session.run。
- `动作槽×`：NPU 实际计算的动作槽 / 真实合法动作；越接近 1 越少 padding。
- `长历史CPU回退`：历史超过 `--npu-history` 的累计次数。
- `异步learner 待更新`：CPU learner 尚未消费的梯度更新预算。如果长期快速增长，说明 CPU learner 跟不上 NPU 数据生成速度。

优化目标应该是 **更高的决策/s**，不是强行让任务管理器中的 NPU 利用率等于 100%。减少无效 padding 后，NPU 利用率甚至可能下降，但每秒真实决策数反而更高。

## 4. 数学正确性检查

f5 没有修改四个 Q 网络的任何参数 shape，也没有修改 reward、合法动作、状态特征或训练目标。

开发回归中使用相同随机权重，在 2,477 个真实牌局决策上比较：

```text
CPU 原始前向
vs
128-history + 16-action 分块 fixed actor
```

最大 Q 值绝对误差：

```text
5.12e-8
```

属于 FP32 运算顺序误差。

因此动作分块和较小 NPU 固定 envelope 不改变网络数学含义。

## 5. 模型结构与参数量

```text
bid             994,201
landlord        994,201
landlord_down   994,201
landlord_up     994,201
-----------------------
总计          3,976,804
```

四个网络完全独立。

每个网络：

```text
state 264 -> 344 -> residual 344
history event 68 -> 160 -> position -> residual MLP x2
history summary = attention + masked mean + last
fusion -> 344 -> residual 344
action 57 -> 160 -> residual 160
[state + action] -> 344 -> 160 -> Q value
```

激活仍是：

```python
GELU(approximate="tanh")
```

## 6. 三席独立胜率

日志中的三项统计“真正先出完牌结束该局的座位”：

```text
地主先出完       -> 地主 +1
地主下家先出完   -> 地主下家 +1
地主上家先出完   -> 地主上家 +1
```

最近 200 局三项合计约 100%。

这只是观察三席独立终结能力。训练 reward 仍然遵守斗地主团队目标：任一农民获胜时，两位农民都得到农民阵营正回报。

## 7. Surface Pro 11 环境

固定使用：

```text
onnx                 1.23.1
onnxruntime          1.27.0
onnxruntime-qnn      2.6.0
```

QNN 2.6.0 官方发布说明给出的测试组合也是 ORT 1.27.0，并对应 QAIRT 2.50.40。

安装/修复：

```powershell
python -m pip uninstall -y onnxruntime onnxruntime-qnn
python -m pip install --upgrade -r client/requirements.txt
python -c "import onnxruntime as o, onnxruntime_qnn as q; print(o.__version__, q.__version__)"
```

应看到：

```text
1.27.0 2.6.0
```

强制 NPU 检查：

```powershell
python train.py --backend npu
```

成功以后平时直接：

```powershell
python train.py
```

## 8. 训练默认参数

```text
replay / 子模型          4096
batch                    64
updates per game         4
loss                     SmoothL1 / Huber
epsilon                  0.20 -> 0.05
epsilon 衰减             2,000,000 决策
日志频率                 10 秒
保存频率                 60 秒
NPU 权重同步             60 秒
NPU action batch         16
NPU history envelope     128
NPU perf mode            burst
NPU RPC latency          100 us
异步 CPU learner         NPU 模式默认开启
```

## 9. 常用命令

正常自动选择：

```powershell
python train.py
```

强制 NPU：

```powershell
python train.py --backend npu
```

性能 A/B：

```powershell
python train.py --npu-action-batch 8
python train.py --npu-action-batch 16
python train.py --npu-action-batch 32
python train.py --npu-history 128
python train.py --npu-history 160
python train.py --no-async-learner
python train.py --threads 8
python train.py --threads 10
python train.py --threads 12
```

排障显示 QNN warning：

```powershell
python train.py --npu-verbose
```

其它：

```powershell
python train.py --games 1000
python train.py --seconds 600
python train.py --buffer 8192
python train.py --fresh
python local.py
python client/qoj_cli.py
```

## 10. 续训兼容

f5 模型 schema 没变，仍然是 schema 3。

旧版已经训练数千局时，把配套的：

```text
client/models/current.pt
training.pt
```

放进 f5 相同位置即可直接续训。

f5 的训练 checkpoint schema 升为 4，只是额外保存后台 learner 的独立随机状态。f5 可以读取训练 schema 2/3/4；一旦 f5 保存后，新的 `training.pt` 使用 schema 4。

## 11. 文件说明

```text
client/models/current.pt   已训练模型；发布时保留
training.pt                optimizer/replay/续训状态
.npu_cache/                自动 ONNX/QNN actor 与诊断文件
*.train.lock               训练锁
__pycache__/               Python 字节码缓存
```

发布游戏时只需要训练后的 `current.pt`；无缝续训才需要 `training.pt`。

## 12. 技术参考

- ONNX Runtime QNN Execution Provider：
  https://github.com/onnxruntime/onnxruntime-qnn/blob/main/docs/execution_providers/QNN-ExecutionProvider.md
- QNN EP 2.6.0 release notes：
  https://github.com/onnxruntime/onnxruntime-qnn/releases/tag/v2.6.0

## f6 learner 极致常数优化

f6 不改变四个约 994k 子模型、reward、合法动作、Huber loss 或默认 `4 updates/game`。主要变化仅在训练调度：

- 四个角色使用四路独立 CPU learner，可并行 forward/backward/AdamW；参数锁按角色拆分。
- Windows ARM64/NPU 主机自动默认每路 1 个 PyTorch intra-op 线程，避免四个小模型发生线程池过度订阅。可用 `--threads 1/2/3` 实测；这里的 threads 是“每路”而不是总线程数。
- AdamW 明确使用 `foreach=True`，梯度裁剪也使用 foreach 快路径。
- replay 采用 16 步历史长度分桶采样：先均匀选 anchor，因此每条 replay 的边际抽样概率保持一致，再在同长度桶内组成 batch，减少 `pad_sequence` 后无效历史计算。
- 日志新增 `learner X 更新/s, 预算 Y/s`。稳定训练要求 X >= Y；短时波动允许。
- 默认 `--max-pending 256`：如果 learner 暂时追不上，rollout 会等待，而不是无限堆积更新欠账。不会通过降低 `updates-per-game` 假装加速。
- f6 可以读取 f5/schema 4 的 `training.pt` 与 `current.pt`。升级时无需删除已训练权重和 replay。

推荐 Surface X Elite 先直接运行 `python train.py`。若 `learner 更新/s` 长期明显高于预算，可尝试 `--threads 2`；若反而下降就恢复 `--threads 1`。不建议把 `--max-pending` 设为 0。
