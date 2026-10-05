# QOJ 斗地主 simple：在线客户端、纯训练与全屏本地对战

Windows 11 / Linux，CPU 模型，三个 Python 文件。`client/qoj_cli.py` 负责原来的在线单局、九局比赛、AI 提示/托管和共享规则/模型；根目录 `train.py` 只训练；根目录 `local.py` 只打一局一人两人机的本地对战。

**`python local.py` 立即发牌开局，没有大厅或聊天区。** 界面沿用 QOJ CLI 的全屏动态刷新、日志、牌桌、记牌器和底部手牌/输入框；正常结束自动退出全屏，并在普通终端打印完整日志。

模型、规则、权重格式与上一版兼容。包内已有预训练模型，不需要另外下载代码或权重。本次改动没有额外训练模型；新增 QOJ 收益修正层初值仍为零，未证明达到特定真人水平。

三个入口可以分别启动。`local.py` 不启动、不写入训练；`train.py` 不打开人机界面。可以在两个终端同时运行，它们通过原子保存的模型文件共享最新权重。在线客户端原有命令继续使用。

## 目录与每个文件的作用

| 路径 | 功能 | GitHub 是否保留 |
|---|---|---|
| `README.md` | 唯一说明：安装、在线/本地命令、训练、存储、迁移、来源 | 保留 |
| `.gitignore` | 排除虚拟环境、缓存、训练快照、临时文件 | 保留 |
| `train.py` | 纯 CPU 自博弈训练、恢复、定时保存和模型发布 | 需要训练时必须 |
| `local.py` | 一人两人机，全屏动态 UI，单局自动结束和完整日志 | 需要本地对战时必须 |
| `client/qoj_cli.py` | 在线 CLI、九局比赛、AI、共享规则/模拟器/模型/保存、客户端打包 | 必须 |
| `client/requirements.txt` | 唯一 pip 清单，三个入口共享 | 必须 |
| `client/LICENSE.txt` | 合并 GPL-3.0 和 DouZero Apache-2.0 全文及署名 | 分发时必须 |
| `client/models/current.pt` | 全部角色的可用 CPU 权重 | 必须，或另提供兼容模型 |

压缩包共八个静态文件，三个 `.py`、一份 README。训练数据目录 `train/checkpoints/` 首次训练时自动生成；目录中没有另一个训练入口。无测试目录、升级脚本或重复文档。模型和共享逻辑仍只有一份。

## 最快使用方法

在解压后的 `qoj-ddz-simple` 根目录打开终端，首次安装：

```powershell
python -m pip install -r client/requirements.txt
```

立即打一把本地斗地主：

```powershell
python local.py
```

只训练模型，默认 4 个 CPU 计算线程、每 10 分钟完整保存并发布：

```powershell
python train.py
```

在线 QOJ 客户端：

```powershell
python client/qoj_cli.py
```

三个命令按需选择，不需要依次执行。若要边训练边玩，另开终端运行 `local.py`。本地对战默认读取最新的 `client/models/current.pt` 或 `train/checkpoints/live.pt`；也能在未开始训练时直接使用随包模型。同一快照目录只启动一个训练进程。

旧 `python train/train.py` 和 `client/qoj_cli.py --train` 入口已移除，改用 `python train.py`。训练现在默认只做自博弈，不需要 `--headless`。

## 环境与 pip 库

建议使用 **64 位 Python 3.11 或 3.12**。在线客户端原代码支持 Python 3.10+，整套模型工程还需要对应系统的 PyTorch wheel。

| 库 | 用途 |
|---|---|
| `torch==2.10.0+cpu` | CPU 推理、梯度训练、权重存储；依赖清单包含官方 CPU wheel 索引 |
| `numpy>=1.26,<3` | 手牌编码与训练数组 |
| `curl_cffi>=0.16.3,<0.17` | 在线 HTTP/TLS 请求 |
| `beautifulsoup4>=4.12,<5` | 网页大厅和初始牌局解析 |
| `prompt_toolkit>=3.0.48,<4` | 终端界面、输入框、日志和详情页 |
| `playwright>=1.51,<2` | 在线浏览器验证回退；已放入同一份依赖清单 |

模型始终使用 CPU。你的 Snapdragon Surface 推荐使用原生 ARM64 Python；用 x86-64 Python 时会经过 Windows 模拟。无需 CUDA、Adreno/Hexagon SDK、g++、Node、ONNX 或自行编译。

Windows 11 x86-64/ARM64、Linux x86-64/ARM64 都需要匹配 Python、系统架构的依赖；Linux wheel 还受 glibc 版本影响。浏览器弹窗需要图形桌面，仅 SSH 环境可离线训练，在线使用 HTTP 模式。实体 Surface 的速度需要在本机观察，不能用别的电脑的吞吐量代替。

如使用 Playwright 自带 Chromium，另安装浏览器组件：

```powershell
python -m playwright install chromium
```

这条命令不是安装 Python 库，不能用 `pip install` 代替。已有 Edge/Chrome 时，客户端会先尝试现有浏览器。离线训练不需要浏览器组件、Cookie 或网络。

## 可选：虚拟环境

Windows PowerShell，直接执行环境中的 Python，无需修改执行策略：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r client/requirements.txt
.\.venv\Scripts\python.exe client/qoj_cli.py
```

训练时把最后一条换为 `.\.venv\Scripts\python.exe train.py`。

Linux：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r client/requirements.txt
.venv/bin/python client/qoj_cli.py
```

训练时把最后一条换为 `.venv/bin/python train.py`。不使用虚拟环境且系统只有 `python3` 时，将示例中的 `python` 换成 `python3`。迁移系统时重新创建虚拟环境，不要复制 `.venv/`。

## 在线登录与界面

启动后输入 `__Host-UOJSESSID` 的值，也接受 `__Host-UOJSESSID=值` 或完整 Cookie 字符串，只提取这个字段。输入不回显；程序不把 Cookie、CSRF token 或浏览器会话保存到文件。

推荐 Windows Terminal 或 UTF-8 Linux 终端，窗口约 100 列 × 30 行。普通终端运行，不依赖 curses。小窗口会缩短日志/聊天区域，完整内容可用 `/log`、`/chat` 查看。

大厅展示账号资料和 Rating；输入 `1` 单局匹配、`2` 九局记分比赛。排队自动轮询并进入对局。`/cancel` 取消排队，`/refresh` 刷新资料、token 和状态，`/quit` 退出。重新运行会尝试恢复账号正在进行的对局。退出程序不会结束服务器对局，服务器超时和托管规则仍生效。

牌桌显示日志、聊天、阶段、当前行动者、底分、倍率、炸弹数、底牌、记牌器和自己的手牌。座位按出牌顺序旋转，自己排在最后；`>` 标记行动者，`[托管]` 标记服务器托管。简洁日志中的 `jump` 是过，`*` 表示服务器托管操作。界面不显示倒计时和日志时间，刷新保留正在输入的内容。

对局中不显示未公开的对手手牌；在线终局可展示服务器公开的剩余手牌。记牌器按整副牌扣除公开已出牌和自己的手牌，不把未知牌分给某个对手。AI 仅使用自身手牌和公开信息，过滤完整手牌、公平性牌堆、Cookie 等字段。本地练习始终隐藏两个 bot 的具体手牌。

## 牌面与在线命令

| 输入 | 行为 |
|---|---|
| `789XJQKA`，`/play 3334` | 出牌，根据自己的实际卡片 ID 选牌 |
| `-` | 出牌阶段过，叫分阶段不叫 |
| `0` / `1` / `2` / `3` | 叫分阶段不叫或叫相应分数；出牌阶段 2、3 是牌 |
| `X` / `T` / `10` | 10；支持大小写、空格和逗号 |
| `S` / `小王`，`D` / `大王` | 两张王；`SD` 为王炸 |
| 普通非牌面文字 | 在线聊天，最多 60 个 UTF-16 字符 |
| `/say 333` | 强制把牌面文字作为聊天发送 |
| `牌面#编号`，`/choose 编号` | 选择歧义牌型；状态变化后旧选择失效 |
| `/ai` | 轮到自己时显示本地模型的叫分/出牌建议，不提交动作 |
| `/bot on` / `/bot off` | 开启/关闭本地模型托管，默认关闭 |
| `/hint` | 原 CLI 的本地规则建议，不调用模型或服务器 AI |
| `/auto on` / `/auto off` | 开启/关闭 QOJ 服务器托管 |
| `/mute`，`/sort` | 屏蔽/恢复他人聊天；切换手牌排序 |
| `/log`，`/chat` | 查看全部出牌日志/聊天 |
| `/fair` | 发牌承诺详情；终局 SHA-256、牌序和随机盐核验 |
| `/initial` | 终局查看最后一次发牌的三人开局手牌和底牌 |
| `/score` / `/scores` | 九局比赛分表、最终名次及 Rating |
| `/again` | 单局结束再匹配单局；整场比赛结束再匹配记分比赛 |
| `/home` | 单局/整场结束后回大厅 |
| `/browser`，`/refresh` | 打开/重新验证浏览器；刷新资料和状态 |
| `/help`，`/close` / Esc | 操作说明；关闭详情，返回最新日志/聊天 |
| F2 / F3，PgUp / PgDn | 翻阅日志/聊天；滚动详情页 |
| `/quit` / Ctrl-C | 退出；排队时尝试取消排队 |

纯牌面字符串始终是出牌尝试，缺牌、不合法、管不上或没轮到自己会显示错误，不会意外发成聊天。多种合法解释先列出选项，用户选择后才提交。

`/bot on` 会关闭自己的服务器托管；`/auto on` 会关闭本地 bot。bot 提交前同步服务器状态并检查局号、版本、行动者和动作合法性。牌局变化则丢弃旧决策。写操作网络超时导致结果不确定时暂停 bot，不自动重发；先 `/refresh` 核对，实际出牌状态变化后可重新开启。

## 九局记分比赛

比赛代码已合并，不需要 `qoj_match.py`、`qoj_ai.py` 或 `upgrade.py`。大厅输入 `2` 即可。

牌局中显示比赛编号、当前局数、累计分，`/score` 查看九局分表。每局结束显示分表和结算，等待约五秒后，按服务器提供的 `next_game` 自动进入下一局。下一局编号尚未出现或加载失败时保留分表继续轮询，不自行创建牌局或重复匹配。

第九局结束展示服务器给出的最终名次、总分和 Rating 变化。分表按用户名对应数组，再按座位顺序显示，自己排最后；未结算局显示 `—`。

局间不提交出牌或聊天，整场未结束不能 `/again`、`/home`。跨局清空旧输入、牌型选择和聊天游标，避免上一局内容发到下一局；同一局刷新仍保留正在输入的内容。已开启的本地 bot 会沿用到下一局。

## Cloudflare 与连接

默认 `--transport auto` 先用 curl-cffi 发送 Chrome 兼容请求，遇验证时尝试真实浏览器。需要点击验证时，在弹窗中完成并保持窗口打开；游戏操作仍在 CLI 输入。后续使用该页面的同源 fetch，不把浏览器 clearance 复制到另一套 HTTP 客户端。客户端阻止页面的斗地主业务脚本，避免网页和 CLI 双重轮询。

```powershell
python client/qoj_cli.py --transport browser --browser msedge
python client/qoj_cli.py --transport http
python client/qoj_cli.py --demo
```

Windows 优先尝试 Edge、Chrome，再尝试 Chromium；Linux 优先 Chrome，再 Chromium。Linux 缺浏览器系统依赖时可按提示运行 `python -m playwright install --with-deps chromium`，可能需要管理员权限。

`http` 模式不自动打开浏览器，适合不需要弹窗验证的终端 Linux。`--demo` 是无需 Cookie、无需网络的界面演示，可查看 `/ai`，不提交动作。其余参数：`--browser auto|chromium|chrome|msedge`，`--timeout 3–60`，默认超时 12 秒。

不保证绕过 Cloudflare。持续被拒绝时可用 `/browser` 重试，或先恢复普通浏览器的正常访问。仅 SSH 环境无法完成弹窗交互。

游戏状态约每 0.9 秒轮询，排队约 1.5 秒，终局约 3 秒；失败后退避。写请求超时不自动重发，应核对随后收到的状态。整个本地训练流程不调用这些在线接口。

## 本地单局：local.py

启动即发牌，默认你在 1 号座位，两名人机共用同一个最新模型。默认每次重新随机发牌，并均匀随机决定先叫者；按规则由叫分确定的地主首先出牌。重发牌继续按模型记录的规则随机/固定先叫者。

UI 使用与 QOJ CLI 相同的全屏终端布局和颜色：上方出牌日志，中间阶段/行动者/底分/倍率/底牌/记牌器和两家公开牌桌，自己的手牌固定靠下，最后是状态和输入框。去掉聊天区、匹配大厅和比赛详情；模型提示显示在状态区，不盖住手牌。轮到人机时自动行动，并短暂停留以便阅读；刷新保留正在输入的内容。

未出的对手手牌始终隐藏，包括终局日志。模型只接收行动者自己的手牌和公开信息。日志中的 `*` 表示模型自动操作，`[托管]` 显示人机或你的本地托管状态。

| 本地输入 | 功能 |
|---|---|
| `0` / `1` / `2` / `3`，`-` | 叫分；出牌阶段 `-` 为过，2、3 为实际牌 |
| `789XJQKA`，`/play 3334` | 出牌；支持大小写、空格、逗号、`10/T/X` 和大小王 |
| `牌面#编号`，`/choose 编号` | 选择歧义牌型 |
| `/ai` | 轮到自己时给出当前模型建议，不自动出牌；估值不是胜率 |
| `/bot on` / `/bot off` | 开启/关闭你自己的模型托管；默认关闭 |
| `/sort` | 切换自己的手牌升降序 |
| `/help` | 在状态区显示本地操作说明 |
| F2 / Esc | 日志向前翻阅 / 回到最新日志 |
| `/quit` / Ctrl-C / Ctrl-D | 提前退出，并打印已发生的日志 |

本地没有聊天，也没有 `/again`、`/home`、`/auto`、`/hint`、`/refresh`、`/browser`、`/score`、`/save`、`/status` 等在线/训练命令。正常打完后自动退出，不等待输入、不自动开始第二把。终端打印发牌、你的初始手牌、叫分、公开底牌、全部出牌/过牌、托管变化与结算；不会打印对手的隐藏手牌。

可选参数只有：

| 参数 | 含义 |
|---|---|
| `--model 路径` | 固定读取某个模型；不在 current/live 两个默认路径间切换 |
| `--seat 1` | 你的座位，可选 1/2/3，默认 1 |
| `--seed 数字` | 需要复现某次发牌时使用；默认不设置，保持随机 |

例如 `python local.py --model client/models/current.pt`。不指定模型时，每次需要决策都会选择 current/live 中较新的一份，并热加载更新。使用自定义训练输出目录时，可显式 `--model 该目录/live.pt`。

## 纯训练：train.py

默认持续 CPU 自博弈，没有终端牌桌、人类输入、聊天或网络请求。人机对战的结果不会加入训练。启动、周期发布和正常退出会打印训练局数、决策数、更新数及速度；每约 30 秒另打印一次进度并刷新 `train/checkpoints/status.json`。

| 参数 | 默认值 / 含义 |
|---|---|
| `--threads 4` | 4 个 CPU 计算线程，不代表 4 个 bot |
| `--batch 128`，`--buffer 2000` | 梯度批量；每角色回放容量 |
| `--min-samples 32` | 每角色开始更新前的最少样本数 |
| `--epsilon 0.05` | 自博弈探索概率 |
| `--base-lr 1e-5`，`--new-lr 1e-4` | 预训练层、新层的学习率；恢复时沿用优化器状态 |
| `--save-interval 600`，`--live-interval 30` | 完整保存/客户端发布、练习用 live 权重更新间隔，秒 |
| `--keep 6` | 保留最近 6 份时间戳快照 |
| `--games N`，`--seconds N` | 本次局数/时间上限；0 为不限 |
| `--seed 20261004` | 初始随机种子；恢复时沿用快照随机状态 |
| `--output train/checkpoints` | 训练数据保存目录 |
| `--deploy-model client/models/current.pt` | 加载并定时发布的客户端权重路径 |
| `--fresh` | 不恢复 latest.pt，从当前客户端权重另起训练 |
| `--force-after-redeals 3`，`--redeal-start random` | 连续流局强制叫分规则和重发先叫者规则 |
| `--publish 路径` | 仅发布某个已有模型/快照，不训练 |
| `--import-pretrained 目录` | 仅从本地 AlphaDou checkout 重新导入初始模型 |

只训练八小时：

```powershell
python train.py --threads 4 --seconds 28800
```

再次运行自动恢复 `latest.pt`。按 Ctrl-C 停止并保存。突然断电/强制结束只能恢复到最近快照；回放缓存和未结束牌局重新积累。想查看当前统计，可打开 `train/checkpoints/status.json`；本地 UI 不显示训练状态。

## 规则与模型原理

54 张牌，各家 17 张、底牌 3 张。随机选先叫者，按顺序各叫一次，不叫或叫更高的 1/2/3；3 分立即成为地主，否则最高叫分者获得公开底牌并先出。

默认连续流局三次后，**第四次发牌**的最后一家必须叫分。若要第三次发牌强制叫，用 `--force-after-redeals 2`。重发默认重新随机先叫者，用 `--redeal-start same` 保持原先叫者。改规则时另选 `--output`，避免恢复不同规则的训练快照；在线始终听从服务器状态。

单张、对子、三张、三带一/对、顺子、连对、飞机及同数量单/对翼、四带二/两对、炸弹、王炸沿用原 CLI 判断。主体不含 2 和王，带牌不与主体同点，不允许炸弹或王炸作为带牌。手动出牌保留歧义选择；裸飞机排前，同类较大的解释排前。

倍率初始 1，每个炸弹或王炸加 1，春天/反春再加 1。地主胜得 `2 × 底分 × 倍率`，每位农民输一份；农民胜反之。春天是地主胜且农民均没出过牌；反春是农民胜且地主只出过第一手牌。

出牌沿用三个位置的 DouZero LSTM 和输入编码，新增 120 维候选/规则特征的收益修正层。叫分使用冻结的监督手牌强度网络，加可训练收益修正层；初始按强度阈值选分。

**总参数 6,299,336，可训练参数 4,564,615，发布模型约 25.2 MB（24.1 MiB）。** 模型格式仍为 schema 1 的 PyTorch CPU `state_dict`，脚本拆分不改变参数键或维度，可以继续读取上一版模型和训练快照。

自博弈枚举合法动作，模型评分并探索，用终局实际得分作为 Monte Carlo 训练目标。叫分记录学习行动座位的最终收益；地主收益除以 6、农民除以 3，梯度裁剪为 5。底分、倍率、出牌次数作为输入，可学习炸后输牌的代价，不依靠巨大牌力分类讨论。

## 保存、热更新与模型接入

| 自动生成路径 | 用途 |
|---|---|
| `client/models/current.pt` | 全角色 CPU 权重和元数据，每 600 秒更新；在线下一次决策热加载 |
| `train/checkpoints/latest.pt` | 权重、Adam、局数/决策数、Python/Torch 随机状态；自动恢复训练 |
| `train/checkpoints/checkpoint_*.pt` | 每次完整保存的时间戳快照，默认保留 6 份 |
| `train/checkpoints/live.pt` | 约每 30 秒更新，供 local.py 使用 |
| `train/checkpoints/reference.pt` | 首次训练启动时保存的初始权重，用于回退 |
| `train/checkpoints/status.json` | 最近保存的训练统计 |

启动、正常退出和 Ctrl-C 退出也完整保存。单个发布文件包含全部角色；先写临时文件、同步，再原子替换，避免客户端读取写到一半的权重。

不需要训练完再手工转换 ONNX 或复制若干角色文件：`train.py` 直接更新 `current.pt`，共享主文件的 `Agent` 读取它，`BoundedEngine` 提供有时限的建议，在线 `/ai` 和 `/bot` 已接好。

**任意一次发布后，复制整个 `client/` 就能带走当前软件。** 目标电脑还要安装 Python 和依赖；在复制的目录中执行 `python -m pip install -r requirements.txt`、`python qoj_cli.py`。不要复制缓存或 `.writing-*` 临时文件。

也可以在项目根目录一条命令打包当前客户端：

```powershell
python client/qoj_cli.py --package
```

生成只用于在线客户端的 `qoj_bot_client.zip`，包含唯一一份 README，以及 `client/` 内的主文件、依赖、许可和当前模型；不包含训练快照、train.py 或 local.py。解压后按上面的在线命令运行。需要自定义 ZIP 路径时，用 `--package my_bot.zip`。

## 决策时限与训练时间

本地引擎默认 2 个计算线程，决策预算 4 秒，包含工作进程就绪等待、模型加载、动作枚举、编码和进程通信。超过预算会终止推理进程，返回合法保底：普通叫分不叫，被强制时最低合法分；跟牌时过；自由出牌时最小单张。操作系统调度不能提供严格实时保证，QOJ 网络请求时间也不属于本地推理时间。

Surface 的训练速度应观察训练终端或 `train/checkpoints/status.json` 的 `decisions_per_second`。100 万次决策大约需要 `1000000 / 决策每秒 / 3600` 小时。没有已验证的 CPU 训练时长能保证达到“打过几百把的真人”水平；自博弈也不保证每次更新都更强。包内预训练出牌权重用于减少从零学习的成本。

## 模型回退与上一版迁移

停止训练后，把某个旧快照发布给客户端：

```powershell
python train.py --publish train/checkpoints/reference.pt
```

发布只切换客户端模型；下次默认训练仍恢复 `latest.pt`。要从切换后的模型另起训练：

```powershell
python train.py --fresh --output train/checkpoints_new
```

已有上一版训练数据时，用旧 `client/models/current.pt` 覆盖本包同名模型，并复制旧 `train/checkpoints/` 到同名目录，再正常启动。旧版拆分模块、工具和测试文件不需要复制。不要让新旧两个训练器同时写同一个目录。

## GitHub 上传

上传本包解压后的这八个静态文件即可，保持目录关系。模型必须一起保留，或另提供兼容模型下载；模型当前约 24.1 MiB，可用普通 Git 保存。`.gitignore` 自动排除训练快照、缓存、虚拟环境和 ZIP。

```bash
git init
git add .
git commit -m "Add compact QOJ CLI and offline CPU trainer"
git branch -M main
```

在 GitHub 建好仓库后配置 `origin` 再推送。上述命令依次创建本地仓库、加入静态文件、创建首次提交、把主分支命名为 `main`。不要仅把整个 ZIP 当作仓库唯一文件，也不要上传 Cookie 或虚拟环境。

`current.pt` 是被跟踪的发布模型；日后训练改变它，可按自己的需要提交新版。运行产生的快照不是启动客户端的必要文件。

## 来源、致谢与许可

本工程包含源码和权重，按 GPL-3.0 分发，并保留 DouZero 的 Apache-2.0 许可。两份许可全文已合并在 `client/LICENSE.txt`，独立复制客户端时一并携带。用户提供的原 QOJ 文件保留原作者署名，不主张其所有权。

| 来源 | 复用内容 |
|---|---|
| 用户提供的原 QOJ CLI 和比赛扩展 | 原登录、通信、终端 UI、规则、聊天、公平性、单局及比赛流程，现合并进主文件 |
| [QOJ 大厅](https://qoj.ac/games/doudizhu)、[游戏脚本](https://qoj.ac/js/games/doudizhu.js)、[规则脚本](https://qoj.ac/js/games/doudizhu-rules.js)、[大厅脚本](https://qoj.ac/js/games/doudizhu-lobby.js) | 协议、牌型与比赛流程参照，实际结算以服务器为准 |
| [DouZero](https://github.com/kwai/DouZero)、[ICML 2021 论文](https://proceedings.mlr.press/v139/zha21a.html) | 三个位置的 LSTM 定义和公开信息输入编码，Apache-2.0 |
| [AlphaDou](https://github.com/RuBP17/AlphaDou) | `baseline/test/` 的 DouZero-ADP 权重、监督叫分 Net2 与权重，GPL-3.0；Net2 原署名 Vincentzyx |
| [curl_cffi](https://curl-cffi.readthedocs.io/en/stable/impersonate/targets.html) | 浏览器兼容 HTTP 请求 |
| [Playwright](https://playwright.dev/python/docs/browsers) | 浏览器启动、验证和同源 fetch |
| [PyTorch](https://download.pytorch.org/whl/cpu/torch/)、NumPy、Beautiful Soup、prompt_toolkit | CPU 学习、数组、网页解析与终端组件 |

DouZero 固定 commit：`718a5c920bf3361e34178a38f3b80458e176b351`。提取 `douzero/dmc/models.py` 的 `LandlordLstmModel`、`FarmerLstmModel`，以及 `douzero/env/env.py` 的纯观察/编码函数和常量；保留网络维度与特征顺序。

AlphaDou 固定 commit：`13e740c08c3b653c2bef6ca345fc8fa6adc7d362`。提取 `baseline/SLModel/BidModel.py` 的 `Net2`。出牌权重来自其 DouZero-ADP baseline，不是 AlphaDou 更新的 ResNet/Card Model。

新增部分包括 QOJ 模拟器、完整动作枚举、公开信息适配、收益修正、叫分微调、独立推理进程、离线对战和原子保存/发布。本版保留合并的共享引擎和在线功能，将纯训练移到 train.py，将无聊天的全屏本地单局移到 local.py；保留原网络、权重格式、规则和在线协议。保存/导入仍属于训练入口，客户端打包仍属于在线入口。

原始权重 SHA-256：

| AlphaDou 相对路径 | SHA-256 |
|---|---|
| `baseline/test/landlord.ckpt` | `0013495862cd853d6a9f108a27043cde0b0a2c6ee15e9fa649f92348d2c5bbc7` |
| `baseline/test/landlord_down.ckpt` | `ef3ea2fbfe5c0286bd9fd27abc40daf702f939b1cd2beddf96ad0fde8b4fb4c7` |
| `baseline/test/landlord_up.ckpt` | `bdb6f7d820ac2cfc1d73cf985fa36c3b0a44eb15b3ca4fd35211d2ebe7bfbe8f` |
| `baseline/SLModel/bid_weights_new.pkl` | `320e9d96e3aa6f962e27f0b8f5f1d97c15af719bc5cb30ed5423b9c025bb4801` |

随包的 `current.pt` 是这些权重加上零输出修正层和元数据，不是已经完成 QOJ 定制训练的声明。

普通使用完全不需要下面的操作。只有想重新生成初始权重时才 clone：

```bash
git clone https://github.com/RuBP17/AlphaDou.git upstream_alphadou
git -C upstream_alphadou checkout 13e740c08c3b653c2bef6ca345fc8fa6adc7d362
python train.py --import-pretrained upstream_alphadou
```

导入产生 `client/models/imported.pt`，不会覆盖当前发布模型。停止训练后，用 `python train.py --publish client/models/imported.pt` 切换。联网只发生在手工 `git clone` 时，权重导入本身只读本地文件。

## 常见问题

- 提示缺依赖：用运行程序的同一个 Python 执行 `-m pip install -r client/requirements.txt`。
- 提示缺浏览器：运行 `python -m playwright install chromium`，或指定已有 Edge/Chrome。
- 本地决策提示保底：模型尚未就绪、加载失败或超时；下一次会重试启动，具体原因会显示。
- 找不到训练快照目录：开始训练后自动生成。
- 更改规则无法恢复：使用新的 `--output`，或明确使用 `--fresh`。
- 页面结构变化：需要最新大厅/对局 HTML 或脱敏接口响应；去掉 Cookie 和 `_token`。
- 只想在线使用：复制 `client/` 即可，训练数据不需要随软件分发。

原网页动画回放和排行榜浏览不属于这个终端客户端。
