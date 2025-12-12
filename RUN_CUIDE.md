# Gaussian-LIC Coarse-to-Fine 运行指南

## 🚀 快速开始

### 1. 运行示例程序（推荐首先运行）

```bash
# 进入工作空间
cd /path/to/your/catkin_workspace

# 运行coarse-to-fine示例程序
rosrun gaussian_lic coarse_to_fine_example
```

**预期输出**：
```
=== Coarse-to-Fine Sampling Strategy Demo ===

Simulating point cloud data...
Generated 10000 points

[CoarseToFine] Manager initialized successfully
Initializing with coarse-to-fine strategy...
[CoarseToFine] Starting coarse initialization...
[CoarseToFine] Sampled 3000 points from 10000 total points (ratio: 0.300)
[CoarseToFine] Initialized with 0.30w GS (coarse level)

Simulating training loop...
Iteration 500: Applying densification...
[CoarseToFine] Iteration 500: Densification applied (split: 45, clone: 23)
Iteration 500: Current Gaussian count: 3068

Iteration 3000: Applying pruning...
[CoarseToFine] Iteration 3000: Pruning applied, removed 12 Gaussians
Iteration 3000: Current Gaussian count: 3056

=== Demo completed successfully! ===
Final Gaussian count: 3056
Current sampling ratio: 0.300
```

### 2. 运行简单测试程序

```bash
# 运行基础功能测试
rosrun gaussian_lic test_coarse_to_fine
```

**预期输出**：
```
Testing Coarse-to-Fine Implementation...
✓ Parameters created successfully
✓ Manager initialized successfully
✓ Manager retrieved successfully
Current sampling ratio: 0.3
All tests passed!
```

## 🔧 配置coarse-to-fine策略

### 1. 修改配置文件

编辑你的配置文件（如 `config/r3live.yaml`）：

```yaml
# === Coarse-to-Fine 策略配置 ===
enable_coarse_to_fine: true          # 启用coarse-to-fine策略

# 初始采样参数
initial_sampling_ratio: 0.3          # 初始采样30%的点
min_initial_points: 1000             # 最少1000个初始点
max_initial_points: 50000            # 最多50000个初始点

# 密集化参数
densify_grad_threshold: 0.0002       # 梯度阈值
densify_size_threshold: 20.0         # 尺度阈值（分裂vs克隆）
densify_start_iter: 500              # 第500次迭代开始密集化
densify_stop_iter: 15000             # 第15000次迭代停止密集化
densify_interval: 100                # 每100次迭代执行一次密集化

# 修剪参数
prune_opacity_threshold: 0.005       # 不透明度阈值
prune_scale_threshold: 100.0         # 尺度阈值
prune_start_iter: 3000               # 第3000次迭代开始修剪
prune_interval: 100                  # 每100次迭代执行一次修剪

# 其他原有参数保持不变...
```

### 2. 对比配置

为了观察效果差异，你可以准备两个配置文件：

**config_traditional.yaml** (传统方法):
```yaml
enable_coarse_to_fine: false
# 其他参数相同...
```

**config_coarse_to_fine.yaml** (coarse-to-fine方法):
```yaml
enable_coarse_to_fine: true
initial_sampling_ratio: 0.3
# 其他coarse-to-fine参数...
```

## 🎯 运行主程序

### 1. 使用coarse-to-fine策略运行

```bash
# 确保配置文件中 enable_coarse_to_fine: true
rosrun gaussian_lic gs_mapping
```

### 2. 观察关键输出

**启动阶段**：
```
[CoarseToFine] Manager initialized successfully
[CoarseToFine] Starting coarse initialization...
[CoarseToFine] Sampled 3000 points from 10000 total points (ratio: 0.300)
[CoarseToFine] Initialized with 0.30w GS (coarse level)
```

**训练阶段**：
```
Update 0.31w GS per Iter    # 初始阶段高斯点数量较少
[CoarseToFine] Iteration 500: Densification applied (split: 45, clone: 23)
Update 0.38w GS per Iter    # 密集化后点数增加
[CoarseToFine] Iteration 3000: Pruning applied, removed 12 Gaussians
Update 0.37w GS per Iter    # 修剪后移除低质量点
[CoarseToFine] Switched to refinement level 1 at iteration 5000
```

## 📊 性能监控和对比

### 1. 内存使用监控

```bash
# 在另一个终端监控内存使用
watch -n 1 'ps aux | grep gs_mapping | grep -v grep'
```

**预期观察**：
- 启用coarse-to-fine：初始内存使用减少约30-70%
- 传统方法：从开始就使用全部内存

### 2. 训练速度对比

观察每次迭代的时间：
- **初期阶段**：coarse-to-fine应该更快
- **后期阶段**：两种方法速度接近

### 3. 高斯点数量变化

观察 "Update X.XXw GS per Iter" 的数值变化：
- **coarse-to-fine**：从低数值开始，逐渐增加
- **传统方法**：从开始就是高数值

## 🔍 详细效果分析

### 1. 启动时间对比

```bash
# 测量启动时间
time rosrun gaussian_lic gs_mapping
```

### 2. 质量评估

如果有测试数据集，可以运行质量评估：

```bash
# 运行完整的SLAM流程后
# 观察最终的渲染质量和地图精度
```

### 3. 日志分析

保存运行日志进行分析：

```bash
# 保存coarse-to-fine运行日志
rosrun gaussian_lic gs_mapping 2>&1 | tee coarse_to_fine_log.txt

# 保存传统方法运行日志（修改配置文件后）
rosrun gaussian_lic gs_mapping 2>&1 | tee traditional_log.txt

# 对比两个日志文件
diff coarse_to_fine_log.txt traditional_log.txt
```

## 🎛️ 参数调优建议

### 1. 根据场景调整采样比例

**室内场景**（点云密集）：
```yaml
initial_sampling_ratio: 0.2    # 更低的采样比例
```

**室外场景**（点云稀疏）：
```yaml
initial_sampling_ratio: 0.4    # 更高的采样比例
```

### 2. 根据硬件调整参数

**高性能GPU**：
```yaml
densify_interval: 50           # 更频繁的密集化
prune_interval: 50             # 更频繁的修剪
```

**低性能GPU**：
```yaml
densify_interval: 200          # 较少的密集化
prune_interval: 200            # 较少的修剪
```

### 3. 根据质量要求调整

**高质量要求**：
```yaml
densify_grad_threshold: 0.0001  # 更低的梯度阈值
prune_opacity_threshold: 0.001  # 更严格的修剪
```

**快速处理**：
```yaml
densify_grad_threshold: 0.0005  # 更高的梯度阈值
prune_opacity_threshold: 0.01   # 更宽松的修剪
```

## 🐛 故障排除

### 1. 如果程序崩溃

检查日志中的错误信息：
```bash
# 查看最后几行日志
tail -n 50 coarse_to_fine_log.txt
```

常见问题：
- **内存不足**：减少 `max_initial_points`
- **CUDA错误**：检查GPU内存使用

### 2. 如果效果不明显

尝试调整参数：
```yaml
initial_sampling_ratio: 0.1    # 更激进的采样
densify_start_iter: 100        # 更早开始密集化
```

### 3. 如果性能下降

检查参数设置：
```yaml
densify_interval: 200          # 减少密集化频率
prune_interval: 200            # 减少修剪频率
```

## 📈 预期效果总结

### 成功的指标

1. **内存效率**：初始内存使用减少30-70%
2. **启动速度**：初始化时间减少
3. **渐进式质量**：高斯点数量从少到多逐渐增加
4. **智能优化**：自动密集化和修剪操作
5. **日志输出**：详细的策略执行信息

### 观察要点

- 初始阶段的低内存使用
- 密集化操作的自动触发
- 修剪操作移除低质量点
- 多层级细化的自动切换
- 最终质量与传统方法相当或更好

通过以上步骤，你应该能够清楚地观察到coarse-to-fine策略的效果和优势！