# 具身数据后处理 Pipeline

把多相机的人类示范 / 真机采集数据，变成可训练、可审计的 LeRobot 数据集。

[English](README.md) · [架构说明](docs/ARCHITECTURE.md) · [阶段参考](docs/PIPELINE_STAGES.md) · [排错](docs/TROUBLESHOOTING.md)

---

## 为什么需要它

采集只是上半场。下半场没人写进论文：三个相机的时钟对不齐；手部检测器恰好在
抓取那一刻丢帧；单目手部模型输出的"深度"看着像米、其实不是；机械臂夹爪和
人手的量纲不同；IK 悄悄算出一条越过关节限位的轨迹。

通常的结果是一堆脚本、一个"大概能用"的数据集，以及没有人说得清哪几条 episode
真的可以拿来训练。

这个仓库是下半场：一套**可配置、可断点续跑、可审计**的后处理流水线，并在每个
阶段设置显式的质量门。不合格的 episode 不会被悄悄放进去。

## 流程

```text
原始多相机数据
→ Episode 完整性检查          (manifest_raw)
→ 15Hz 因果时间同步            (sync_15hz)
→ 双视角手部姿态识别           (hamer_camera1 / camera3)
→ 双目恢复米制手腕轨迹         (stereo_wrist)
→ 手掌闭合程度 + 真机夹爪先验   (robot_gripper_prior → fuse_gripper)
→ 机器人轨迹重定向与 IK        (robot_ik)
→ 质量检查与 Episode 筛选      (manifest_ik_ready → audit)
→ 打包 LeRobot 训练数据        (pack_lerobot)
→ 生成归一化统计
```

每个箭头都是 JSON 配置里的一个 stage：一条 argv 数组、一个工作目录、若干输入
输出路径、若干可接受返回码。编排层不认识 HaMeR、Nero、Revo2 或 LeRobot，所以换
后端是改配置，不是改代码。

## 设计原则

- **默认只预览。** 不加 `--execute` 时只打印解析后的变量、缺失的前置条件和将要
  执行的完整命令。
- **不经过 shell。** stage 是 argv 数组，含空格的路径安全，实际执行的 argv 会
  原样写进 provenance。
- **原始数据只读。** 筛选通过 manifest 表达，从不删除、移动或重编号 episode。
- **质量是门，不是日志。** `status: PASS/FAIL` 会被下一个阶段消费；双目失败的
  episode 到不了打包环节。
- **弱信号必须标注。** 夹爪相位先验只是遮挡兜底；单目深度不会被说成米制深度。

## 快速开始

```bash
git clone https://github.com/fangdiyu/embodied-postprocess-pipeline.git
cd embodied-postprocess-pipeline
python -m pip install -r requirements.txt
```

编排器本身只依赖 Python 标准库；NumPy / pandas / PyArrow 是表格工具需要的。

```bash
# 列出阶段
python postprocess_pipeline/pipeline.py \
  --config postprocess_pipeline/configs/ego_hamer_stereo_ik.template.json --list

# 用 --set 覆盖路径，默认是 dry-run
python postprocess_pipeline/pipeline.py \
  --config postprocess_pipeline/configs/ego_hamer_stereo_ik.template.json \
  --set source_root=/datasets/my_ego_recordings \
  --set robot_dataset=/datasets/my_robot_demonstrations

# 确认无误后
python postprocess_pipeline/pipeline.py \
  --config postprocess_pipeline/configs/ego_hamer_stereo_ik.template.json \
  --set source_root=/datasets/my_ego_recordings --execute
```

## 不需要任何数据就能跑

demo 会生成一份合成的两 episode 采集数据，然后跑**真实的**流水线：不需要相机、
GPU、ffmpeg 或任何外部模型。

```bash
python examples/make_demo_dataset.py
python postprocess_pipeline/pipeline.py --config examples/demo.pipeline.json --execute
```

预期结果：3 条 episode 通过结构检查，3 条建好 15Hz 时间线，夹爪先验得到
open 0.152 / closed 0.467；融合阶段 2 条 PASS，第 3 条因检出率 51.1% 低于 60%
被判 FAIL，最终 IK 清单只剩 2 条，拒绝原因写进 manifest。

第 3 条是**故意**做成丢帧严重的。demo 要证明的是质量门确实会拦住数据，而且
拦得有据可查。

## 仓库结构

```text
postprocess_pipeline/
  pipeline.py                 通用调度器（仅标准库）
  pipeline.schema.json        配置 schema
  tools/                      manifest / 过滤 / 先验 / 审计
  configs/                    可移植模板 + 真实配置示例
adapters/
  build_ego_no_rokoko_15hz_timeline.py   15Hz 因果时间线
  fuse_hamer_revo2_gripper.py            视觉闭合 + 真机锚点融合
  build_ego_fruit_ik_trajectory.py       米制手腕 → 7-DoF 关节轨迹
  prepare_pi05_ego_fruit_lerobot.py      LeRobot v3 打包
examples/                     合成 demo
docs/                         架构、阶段参考、排错
```

## 哪些东西没有一起开源

编排器、manifest / 审计工具，以及时间线、融合、IK、打包适配器都是完整可跑的。
有三样东西**故意没有内置**，因为它们有各自的许可证和发布节奏：

| 未内置 | 原因 | 接入方式 |
| --- | --- | --- |
| 手部姿态估计（HaMeR / MediaPipe / MANO） | 权重与许可在上游 | `hamer_camera1`、`hamer_camera3`、`stereo_wrist` 三个阶段 |
| 机器人 URDF + IK 求解器 | 与具体机型绑定，常受 NDA 限制 | `--ik-backend-script` |
| LeRobot 打包辅助代码 | 属于 HumanEgo 工具链（**PolyForm 非商业 1.0.0**） | `HUMANEGO_PIPELINE_ROOT`，或自己写一个 packer |

打包适配器采用**惰性导入**：没装 HumanEgo 时 `--help` 仍然可用，缺依赖会给出
一条可操作的错误信息，而不是 `ImportError` 堆栈。

开发时使用的手部姿态与双目参考实现在配套仓库
[`ego-dropper-toolkit`](https://github.com/fangdiyu/ego-dropper-toolkit)。只要满足适配器契约，任何后端都可以接入——流水线不会 import 它。

## 许可

核心编排与原创工具：**LGPL-3.0**。第三方模型、权重、数据集和 SDK 保留各自条款，
见 [NOTICE](NOTICE) 与 [THIRD_PARTY.md](postprocess_pipeline/THIRD_PARTY.md)。

发布前请勿包含私有服务器地址、凭据、硬件序列号、个人绝对路径，以及未取得
参与者同意的采集数据。`build_release.py` 会按"形状"扫描这些内容。

## 引用

见 [CITATION.cff](CITATION.cff)。

## 贡献

欢迎 issue 和 PR，尤其是新的后端（MediaPipe、Franka、UR、xArm、ACT、Diffusion
Policy）。先读 [CONTRIBUTING.md](CONTRIBUTING.md) 和适配器契约。
