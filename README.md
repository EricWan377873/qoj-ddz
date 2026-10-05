# qoj-ddz
QOJ 斗地主 CLI 版本，还有接本地 AI 的

所有东西都是 GPT 写的，本地进行测试过。

final_douzero 是最初始版，我记得是用了 DouZero，但是 DouZero 用的规则毕竟和 qoj 不一样，虽然可能聪明一点。似乎还可以继续训练并微调。

final_begin 好像是 GPT 造了一个新的模型，但是参数规模很小，训练也不快。

final_begin_v2 GPT 进行了一些优化，还在训练阶段加入了一些 NPU 优化，针对我的电脑特化的，所以可能在别的机器上训练更慢。

final_ataraxos 还没造，找时间看看能不能接入。

每个版本训练有关的参数设置看分别的 README，qoj_cli.py 进去后 help 就能看在 qoj 上玩的使用方法。GPT 不会偷你 cookie，如果偷那我的 cookie 是第一个被偷的。
