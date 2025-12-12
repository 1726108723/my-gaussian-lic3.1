# Coarse-to-Fine Sampling Strategy for Gaussian-LIC

本文档介绍了在Gaussian-LIC中实现的从粗到细（coarse-to-fine）采样策略，该策略参考了SplaTAM算法的设计思想。

## 概述

从粗到细的采样策略主要通过以下三个核心机制实现：

1. **初始粗采样**：从输入点云中按比例采样，减少初始高斯点数量
2. **自适应密集化**：基于梯度和尺度信息动态增加高斯点
3. **智能修剪**：移除低质量和冗余的高斯点

## 核心特性

### 1. 初始粗采样
- 支持可配置的初始采样比例（默认30%）
- 设置最小和最大初始点数限制
- 保持点云的空间分布特性

### 2. 自适应密集化
- **分裂（Split）**：对于梯度大且尺度大的高斯点，分裂成两个更小的高斯点
- **克隆（Clone）**：对于梯度大但尺度小的高斯点，直接复制
- 基于可配置的梯度和尺度阈值

### 3. 智能修剪
- 移除不透明度低于阈值的高斯点
- 移除尺度过大的高斯点
- 定期执行以保持模型效率

## 配置参数

在配置文件（如`r3live.yaml`）中添加以下参数：

```yaml
# Coarse-to-fine sampling strategy parameters
enable_coarse_to_fine: true
initial_sampling_ratio: 0.3  # Start with 30% of points
min_initial_points: 1000
max_initial_points: 50000
densify_grad_threshold: 0.0002  # Gradient threshold for densification
densify_size_threshold: 20.0    # Size threshold for split vs clone
densify_start_iter: 500         # Start densification after 500 iterations
densify_stop_iter: 15000        # Stop densification after 15000 iterations
densify_interval: 100           # Densify every 100 iterations
prune_opacity_threshold: 0.005  # Prune Gaussians with opacity < 0.005
prune_scale_threshold: 100.0    # Prune oversized Gaussians
prune_start_iter: 3000          # Start pruning after 3000 iterations
prune_interval: 100             # Prune every 100 iterations
```

## 使用方法

### 1. 初始化Coarse-to-Fine管理器

```cpp
#include "coarse_to_fine.h"

// 创建参数结构
CoarseToFineParams ctf_params;
ctf_params.initial_sampling_ratio = 0.3f;
ctf_params.densify_grad_threshold = 0.0002f;
// ... 设置其他参数

// 初始化全局管理器
initializeCoarseToFineManager(ctf_params);
```

### 2. 使用Coarse-to-Fine初始化

```cpp
// 创建高斯模型
auto gaussian_model = std::make_shared<GaussianModel>(params);

// 使用coarse-to-fine策略初始化
if (params.enable_coarse_to_fine) {
    gaussian_model->initializeCoarseToFine(dataset);
} else {
    gaussian_model->initialize(dataset);  // 传统方法
}
```

### 3. 在训练循环中应用策略

```cpp
for (int iteration = 0; iteration < max_iterations; ++iteration) {
    // 执行优化步骤
    double loss = optimize(dataset, gaussian_model, iteration);
    
    // coarse-to-fine策略会在optimize函数内部自动调用
    // 包括：updateRefinementLevel, adaptiveDensification, pruneGaussians
}
```

## 算法流程

### 初始化阶段
1. 根据`initial_sampling_ratio`从输入点云中采样
2. 为采样点创建初始高斯点
3. 设置初始的尺度、旋转、不透明度等参数

### 训练阶段
1. **梯度计算**：在每次反向传播后计算位置梯度
2. **密集化判断**：
   - 如果梯度 > `densify_grad_threshold` 且尺度 > `densify_size_threshold`：执行分裂
   - 如果梯度 > `densify_grad_threshold` 且尺度 ≤ `densify_size_threshold`：执行克隆
3. **修剪判断**：
   - 如果不透明度 < `prune_opacity_threshold`：标记为删除
   - 如果最大尺度 > `prune_scale_threshold`：标记为删除

### 多层级细化
- 支持多个细化级别，每个级别有不同的采样比例
- 在指定的迭代次数自动切换到下一级别

## 性能优势

1. **内存效率**：初始阶段使用较少的高斯点，减少内存占用
2. **计算效率**：避免处理大量不必要的高斯点
3. **质量保证**：通过自适应密集化确保重要区域的细节
4. **动态优化**：通过修剪策略保持模型的紧凑性

## 示例程序

运行示例程序来了解coarse-to-fine策略的工作原理：

```bash
# 编译
catkin_make

# 运行示例
rosrun gaussian_lic coarse_to_fine_example
```

## 与SplaTAM的对比

| 特性 | SplaTAM | Gaussian-LIC Coarse-to-Fine |
|------|---------|------------------------------|
| 初始采样 | 固定密度 | 可配置比例采样 |
| 密集化策略 | 基于梯度的split/clone | 基于梯度和尺度的split/clone |
| 修剪策略 | 基于不透明度 | 基于不透明度和尺度 |
| 多层级支持 | 有限 | 完全可配置的多层级 |
| 实时性能 | 优化 | 针对SLAM场景优化 |

## 调试和监控

系统提供了详细的日志输出，包括：
- 初始采样统计
- 密集化操作计数（split/clone）
- 修剪操作统计
- 当前高斯点数量
- 细化级别切换信息

通过这些信息可以监控策略的执行效果并调整参数。

## 注意事项

1. **参数调优**：不同场景可能需要调整阈值参数
2. **内存管理**：密集化操作会增加内存使用，需要监控
3. **计算开销**：梯度计算和密集化操作有一定计算开销
4. **收敛性**：过于激进的修剪可能影响收敛性

## 未来改进

1. **自适应阈值**：根据场景自动调整阈值参数
2. **更智能的采样**：基于场景复杂度的采样策略
3. **并行优化**：GPU并行化密集化和修剪操作
4. **质量评估**：集成渲染质量评估来指导策略