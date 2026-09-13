# 重建后端

每个后端是独立 Python 项目，只通过版本化文件协议与核心交换数据。

- `da3/`：已建立适配器骨架和命令行入口，尚未接入模型推理。
- MoGe-3：暂不创建运行环境；需要强对照时再加入。单图几何不能假装具有跨图外参。

DA3 上游要求 NumPy < 2，核心采用 NumPy 2.x。不要把这些项目作为一个 uv workspace 安装，也不要把模型依赖安装进 Blender。DA3 骨架环境的锁文件仅覆盖其当前声明的启动依赖，不代表上游 torch/xformers/CUDA 组合已经可用。

接入顺序见 [技术栈](../docs/decisions/0002-reconstruction-stack.md) 与 [算法接口](../docs/architecture/03-reconstruction-and-refinement.md)。
