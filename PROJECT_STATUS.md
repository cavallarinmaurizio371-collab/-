# PROJECT STATUS

更新时间：2026-08-20（本机验证完成）

## 当前实现

- 路径保护、项目内缓存隔离和本地虚拟环境脚本
- OpenCV 摄像头采集与资源释放
- MediaPipe Hands 21 点及食指几何手势规则
- torchvision SSDLite320 MobileNet V3 Large COCO cup 检测
- Depth Anything V2 Metric Indoor Small 可选加载与明确标注的演示降级
- 深度 patch/bbox 中位数、线性深度校正、内参缩放、2D→3D 反投影
- 食指 PIP→TIP 3D 射线、2D+3D 联合评分、反方向排除、None 阈值
- EMA 与目标切换 hysteresis
- OpenCV 实时 UI、可选 MP4/CSV/JSON 会话记录
- 棋盘格相机标定和模块化测试工具

## 模型与许可

| 模块 | 模型 | 来源 | 许可说明 |
|---|---|---|---|
| Hand | MediaPipe Hand Landmarker float16 | Google MediaPipe Tasks | Apache-2.0；[官方仓库](https://github.com/google-ai-edge/mediapipe) |
| Cup | SSDLite320 MobileNet V3 Large, COCO weights | torchvision / PyTorch | torchvision BSD-3-Clause；COCO 数据条款需单独复核；[官方仓库](https://github.com/pytorch/vision) |
| Depth | Depth Anything V2 Metric Indoor Small (24.8M) | `depth-anything` Hugging Face | Small 模型 Apache-2.0；[模型卡](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf)、[官方仓库许可说明](https://github.com/DepthAnything/Depth-Anything-V2#license) |

## 测试状态

已执行 12 个自动测试、摄像头单帧三模块推理、metric depth 单帧融合，以及 `app.py` 10 帧无窗口冒烟测试。

```text
[PASS] Project path isolation - 外部路径拒绝测试通过，缓存/模型/临时/输出均重定向到项目
[PASS] Camera - index 0 实际读取成功，640x480 BGR 帧
[PASS] Hand landmarks - 官方 Tasks 模型加载及真实摄像头帧推理成功；测试画面当时无手
[PASS] Pointing gesture - 几何正例/异常输入自动测试通过；真人姿势阈值仍需现场调参
[PASS] Cup detection - COCO SSDLite 权重加载及真实帧推理成功；测试画面当时无杯
[PASS] Depth estimation - metric indoor small 实际输出成功，单帧范围 0.328-1.440 m
[PASS] Camera intrinsics - 近似内参加载/缩放与光轴测试通过；真实标定尚未做
[PASS] 2D -> 3D - 主点、坐标轴人工数据测试通过
[PASS] 3D pointing ray - 射线方向、距离、反方向排除测试通过
[PASS] Cup target selection - 2D/3D 阈值、None 与 hysteresis 测试通过
[PASS] Real-time integrated demo - app.py 无窗口连续 10 帧正常运行退出；三杯真人验收待做
```

## 当前已知条件

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，8 GB；当前 PyTorch 2.7.1+cpu，尚未启用 CUDA
- 默认系统 Python 3.13.3 不适合当前视觉依赖；项目使用 Miniconda Python 3.12.4 创建 `.venv`
- 包版本：OpenCV 4.11.0、MediaPipe 0.10.21、NumPy 1.26.4、torchvision 0.22.1+cpu、Transformers 4.57.6、PyYAML 6.0.2
- 真实相机未标定，当前是 `APPROXIMATE_INTRINSICS`
- 深度模型为 metric indoor 版本，但校正样本尚未采集；UI 显示 `METRIC_RAW_UNCALIBRATED`，不能视为高精度真实距离
- CPU 实测：metric 完整单帧约 1.64 FPS；10 帧 approximate-depth 完整 app 平均 1.92 FPS（含首帧模型热身影响）
- 当前摄像头测试画面没有手/杯子，三杯左中右、指向画面外等业务 Case 尚未真人实测，不能据此声称识别准确率达标

## 下一步建议

1. 先运行 Demo，摆放三个外观明显的杯子，按需求中的 8 个 Case 现场验收并用 `--record` 留档。
2. 拍摄棋盘格图片完成真实相机标定，再采集 30/50/70/100 cm 深度校正样本。
3. 若需要更高帧率，在项目内安装与驱动匹配的 CUDA PyTorch；若杯子漏检，再进入业务杯子 detector 微调，而不是更换来源不明模型。
