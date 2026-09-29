# Arbitr3D

基于多视角掩码、几何证据与拓扑复审的 MLLM 引导三维部件分割。

[English](README.md) · [安装说明](docs/installation.md) · [数据与评测](docs/data-and-evaluation.md)

流程包括多视角渲染、SAM 掩码提取、视觉语言模型语义判别、几何与拓扑一致性复审，以及向三维点云的投影和融合。仓库同时保留独立的 PartVerse 指代表达分割（RES）实现。

## 开源内容

包含核心源码、推理/评测入口、知识生成脚本、消融与可视化工具、依赖配置和示例知识文件。不包含数据集、模型权重、实验输出、论文材料或本地密钥。知识文件仅提供少量类别示例，其他类别需要自行生成。

推荐按原项目环境使用 Linux、Python 3.10、PyTorch 2.3.1、CUDA 11.8 与 PyTorch3D。运行前需准备 SAM 权重，以及支持图像输入的 OpenAI 兼容 API 服务。完整安装步骤见 [安装说明](docs/installation.md)。

## 使用步骤

1. 克隆仓库并按安装说明配置环境。
2. 将 `config/config.yaml` 复制为 `config/config.local.yaml`，填写数据目录、SAM 权重、MLLM 服务地址和模型名。
3. 在环境变量中设置密钥。`mllm.api_key_env` 填环境变量名称，不能直接填写密钥；无鉴权本地服务可将该变量设为 `EMPTY`。
4. 在仓库根目录运行推理。

```bash
export OPENAI_API_KEY=EMPTY
export MLLM_BASE_URL=http://localhost:8000/v1
export MLLM_MODEL_NAME=your-vision-model
export CONFIG_PATH=config/config.local.yaml

python infer.py --config "$CONFIG_PATH" \
  --input Chair/179/pc.ply --category Chair --output outputs/chair_179
```

对于自有模型，可传入绝对路径，并通过 `--classes "back,seat,leg,arm"` 指定部件。`--gpu` 使用逻辑编号，例如设置 `CUDA_VISIBLE_DEVICES=2` 后传 `--gpu 0`。结果目录包括预测点云、标签数组、类别映射和中间步骤。网格采样后的点标签不能直接视为原始网格的面标签。

`.env.example` 仅提供环境变量示例，程序不会自动读取 `.env`。Windows PowerShell 可用 `$env:OPENAI_API_KEY = "EMPTY"` 等形式设置变量；完整推理仍以 Linux/CUDA 环境为目标。

## 验证范围

发布前已通过 Python 语法检查、5 项配置与密钥测试，以及原有回归脚本的 6 组检查，覆盖导入、接口、配置兼容、RES 隔离、合成数据指标和查询哈希。当前发布环境缺少 PyTorch3D 与原生 Cut-Pursuit 扩展，尚未进行完整 GPU 推理或重新复现数据集指标。

知识生成和推理会调用配置的模型服务，可能产生 API 费用。模型版本、随机采样、知识文件和 Cut-Pursuit/备用超点实现都会影响结果，请在实验中记录。

## 许可证

项目自有代码采用 [MIT](LICENSE)，第三方代码保留原许可。数据、模型权重与外部服务遵循各自条款，详见 [第三方声明](THIRD_PARTY_NOTICES.md)。
