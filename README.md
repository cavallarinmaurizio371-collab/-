# 单目摄像头手势指向杯子 Demo

使用普通 RGB 摄像头实时检测手部 21 个关键点、食指指向状态和杯子，结合单目深度与相机内参估算 3D 坐标，并输出 `Cup 1 / Cup 2 / Cup 3 / None`。

## 快速开始（Windows PowerShell）

所有安装、模型、缓存、临时文件和输出都留在 `C:\手势识别` 中，不修改系统 PATH，也不做全局安装。

```powershell
cd C:\手势识别
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
.\scripts\run.ps1 --depth approximate
```

`--depth approximate` 可以最快看到完整流程，但显示的是明确标注的近似深度。下载真实模型后使用：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_models.ps1
.\scripts\run.ps1 --depth metric
```

按 `Q` 或 `Esc` 退出；加 `--record` 会将视频、事件和指标保存到 `outputs/session_*`。

## 流程

```text
Webcam RGB
   │
   ├── MediaPipe Hand Landmarker ──> Index 2D + pointing rule
   ├── SSDLite MobileNet V3 (COCO) ──> Cup 2D boxes
   └── Depth Anything V2 Metric Indoor Small ──> Depth Map

Index 2D + Depth + Camera Intrinsics ──> Index 3D
Cup 2D + Depth + Camera Intrinsics ──> Cup 3D
Index 2D/3D ray + Cup 2D/3D positions ──> Cup 1/2/3/None
```

这里的三个感知模块是手部关键点/手势、目标检测、深度估计。3D 坐标与目标选择是几何融合算法，不需要训练第四个网络。

## 独立测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tools\camera_test.py
.\.venv\Scripts\python.exe tools\test_hand.py
.\.venv\Scripts\python.exe tools\test_detector.py
.\.venv\Scripts\python.exe tools\test_depth.py
```

## 相机与深度标定

默认内参按 60° 水平视场角估算，UI 显示 `APPROXIMATE_INTRINSICS`。采集至少 5 张不同姿态的 9×6 内角点棋盘格照片到项目目录后：

```powershell
.\.venv\Scripts\python.exe tools\calibrate_camera.py "outputs\calibration\*.jpg" --cols 9 --rows 6 --square-mm 25
```

结果保存在 `configs/camera_intrinsics.json`。单目 metric depth 仍可能有尺度偏差；用 30/50/70/100 cm 实测样本拟合 `Z_corrected = a * Z_raw + b`，参数保存在 `configs/depth_calibration.json`。程序同时保留 raw 与 corrected 值。

## 科学性与限制

- 单目相对深度只能可靠表达远近顺序；metric 模型试图输出米制距离，但受场景、材质和相机影响，不能等同于深度相机。
- 未下载 metric 模型时使用固定 0.8 m 的演示平面，仅用于验证 2D/3D 数据流，UI 会显示 `APPROXIMATE_CONSTANT_UNCALIBRATED`。
- COCO 检测器可能漏检透明、无把手或外形特殊的杯子。实际场景不稳定时应采集业务图片微调杯子检测器，这是 Phase 2。
- 真实测量前必须完成相机标定和深度线性校正。

## 配置

阈值、2D/3D 权重、EMA、目标切换滞回帧数均集中在 `configs/default.yaml`，没有散落硬编码。坐标系为光心原点，`+X` 向右、`+Y` 向下、`+Z` 向前；现实向上高度为 `-Y`。
