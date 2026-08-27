# Z=0 摄像头平面手势指向专项评测

## 为什么做这个评测

原三杯 Demo 验证完整业务链路；本模块单独评估食指正面朝向摄像头时，关键点、三维方向和预测落点是否稳定。它只调用现有手部、深度、反投影和 `TIP-PIP` Baseline，不修改 `app.py`、杯子检测或原目标选择逻辑。

## Z=0 摄像头平面

相机光心为 `(0,0,0)`，手位于 `Z>0`。现有业务相机坐标为：

```text
+X_camera = 原始图像右
+Y_camera = 原始图像下
+Z_camera = 摄像头朝用户
```

Baseline 射线沿用原系统：

```text
P = TIP_3D
d = normalize(TIP_3D - PIP_3D)
R(t) = P + t*d, t >= 0
```

与 `Z=0` 求交：

```text
t_hit = -Pz / dz
hit = P + t_hit*d
```

`abs(dz)` 小于配置阈值时输出 `NEAR_PARALLEL`；`t_hit<0` 时输出 `POINTING_AWAY_FROM_CAMERA`，程序不会为了得到结果而反转方向。有限交点超出 A4 范围时保留原始数值并标记 `OUT_OF_TARGET_RANGE`，不参与区域准确率或毫米误差。

## Evaluation Plane 坐标

报告使用用户面对电脑时容易理解的坐标：

```text
+X_eval = 用户物理右方 = -X_camera
+Y_eval = 物理上方     = -Y_camera
+Z_eval = 摄像头朝用户 = +Z_camera
```

内部 XYZ 使用米，靶面 Ground Truth、落点和误差使用毫米。

## 镜像如何处理

手部模型、深度、反投影、射线、落点、区域和误差始终使用未经镜像的原始帧。`M` 键只在显示层执行水平翻转。虚拟靶盘及内部评测坐标不会随预览镜像翻转。

## 制作 Ground Truth

1. 以真实摄像头镜头中心为 `(0,0)`，不要使用屏幕中心。
2. 在镜头周围贴九个不遮挡镜头的实体标签。
3. 用户面对电脑，用尺子测量每个标签相对镜头中心的位置。
4. 用户的物理右方为 `+X_eval`，物理上方为 `+Y_eval`。
5. 将实测毫米值填写到 [z0_targets.yaml](/C:/手势识别/evaluation/configs/z0_targets.yaml)。

当前配置采用竖放 A4（210×297 mm），镜头为纸张中心，九点坐标已按用户确认值填写：横向 ±80 mm、纵向 ±120 mm。九区域通过离实测目标最近的区域判断，不按图像像素九等分。

## 运行

完整 Metric Depth：

```powershell
cd C:\手势识别
.\.venv\Scripts\python.exe evaluation\z0_pointing_eval.py --depth metric
```

默认进入 Quick Evaluation：70 cm、CENTER/LEFT/RIGHT/UP/DOWN、每目标 3 次，共 15 次。界面自动提示下一目标，但不会强迫一次完成；按 `Q` 随时结束也会生成当前 Session 报告。

完整 135 次模式：

```powershell
.\.venv\Scripts\python.exe evaluation\z0_pointing_eval.py --depth metric --mode full
```

纯手动模式：

```powershell
.\.venv\Scripts\python.exe evaluation\z0_pointing_eval.py --depth metric --mode manual
```

快速检查界面和交互：

```powershell
.\.venv\Scripts\python.exe evaluation\z0_pointing_eval.py --depth approximate
```

指定本轮实测距离和手出现位置：

```powershell
.\.venv\Scripts\python.exe evaluation\z0_pointing_eval.py --depth metric --distance-cm 70 --hand-position LEFT
```

## 操作

```text
7 / 8 / 9 = LEFT_UP / UP / RIGHT_UP
4 / 5 / 6 = LEFT / CENTER / RIGHT
1 / 2 / 3 = LEFT_DOWN / DOWN / RIGHT_DOWN

SPACE = 准备 2 秒后采集 3 秒
M     = 切换显示镜像（不影响内部结果）
H     = 切换手在画面中的位置标签
D     = 切换测试距离
Q/ESC = 安全退出并生成汇总
```

按 Space 后依次显示 `READY 2s`、`RECORDING 3s`，并提示 `Please point to <TARGET>`。数字键选择目标后自动切换为手动模式。

准备与采集时间、测试距离、稳定性阈值均在 [z0_eval.yaml](/C:/手势识别/evaluation/configs/z0_eval.yaml) 中配置。

## 推荐实验顺序

第一轮先在 70 cm 测试 `CENTER、LEFT_UP、RIGHT_UP、LEFT_DOWN、RIGHT_DOWN`，手分别位于画面 `LEFT、CENTER、RIGHT`。检查关键点是否存在、`dz` 是否为负、交点是否有效、左右是否翻转。

Smoke Test 通过后，再做 50/70/90 cm、九目标、每目标至少五次，共 135 个 Trial。距离必须由尺子实测，不能用模型预测值作为 Ground Truth。

## 输出

每次运行生成：

```text
outputs/z0_eval/session_*/
├─ trials.csv
├─ frames.csv
├─ summary.json
├─ summary.md
├─ failure_analysis.md
├─ region_confusion_matrix.csv
├─ region_confusion_matrix.png
├─ screenshots/
└─ optional_video/
```

`frames.csv` 保留每一帧的原始评测数据；`trials.csv` 使用有效落点中位数作为最终 Trial 预测。正式 Trial 结束时自动保存界面截图。

## 指标含义

- Hand Detection Rate：检测到手的采集帧比例。
- Pointing Recognition Rate：检测到手的帧中，被判定为 POINTING 的比例。
- Valid 3D Direction Rate：获得有效 `TIP-PIP` 三维方向的比例。
- Valid Z=0 Intersection Rate：有效方向中能够沿射线命中摄像头平面的比例。
- Region Accuracy：有效 Trial 中预测九区域与 GT 一致的比例。
- Mean/Median/P90/Max Radial Error：预测落点与实测点的二维毫米误差。
- TIP/PIP/DIP Stability：静止采集期间的二维和三维关键点抖动。
- Direction Jitter：每帧方向相对平均方向的角度标准差。
- Hit Point Jitter：每帧落点相对平均落点的径向 RMS 抖动。

汇总会分别按距离、真实目标和手出现位置统计，并生成九区域混淆矩阵。

界面诊断区同时显示 MCP/PIP/DIP/TIP 的 Z、`TIP-PIP` 原始 dx/dy/dz、方向长度、与摄像头轴夹角、深度顺序、方向质量和 sanity flags。方向质量含 `GOOD / MARGINAL / NEAR_PARALLEL / AWAY / UNSTABLE / INVALID`；`UNSTABLE` 是 Trial 多帧结论。

常见状态：

- `AWAY`：当前方向沿 +Z，射线背离 Z=0 摄像头平面；不会自动反向。
- `NEAR_PARALLEL`：归一化方向的 |dz| 太小，求交会被保护性拒绝。
- `OUT_OF_TARGET_RANGE`：数学交点存在，但落在 210×297 mm A4 范围外。
- `DEPTH_ORDER_INCONSISTENT`：单目深度认为 TIP 比 PIP 更远，正指时通常会导致 dz 为正。

## 当前限制

如果界面显示 `METRIC_RAW_UNCALIBRATED` 或 `APPROXIMATE_INTRINSICS`，报告中的毫米误差同时受到深度未校正和近似相机内参影响。此时结果只能用于验证模型趋势、稳定性和几何链路，不能宣传为最终物理精度。只有完成真实相机内参标定和深度距离校正后，才能称为 `Calibrated Physical Accuracy`。

正指时 TIP/PIP/DIP 容易在图像中重叠，单目深度也可能无法稳定分辨几厘米的关节深度差。本模块会原样记录无效方向、正负 `dz`、关键点丢失和抖动，不做人为修正。

## 带教现场建议

先运行默认 Quick 模式，完成 CENTER、LEFT、RIGHT、UP、DOWN 各 3 次；再用 `H` 切换手位置，重点补采 GT=CENTER 且手位于 LEFT/CENTER/RIGHT/UP/DOWN 的 Trial。退出后直接打开该 Session 的 `summary.md`、`failure_analysis.md` 和混淆矩阵 PNG。
