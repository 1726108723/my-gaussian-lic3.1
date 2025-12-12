/*
 * AdaptiveVoxelManager - 自适应体素管理器
 * 
 * 基于本地几何复杂度维护多分辨率体素网格：
 * - 平坦区域使用大体素减少 Gaussian 数量
 * - 高曲率/高梯度区域使用小体素保留细节
 * - 与现有 coarse-to-fine 策略协同工作
 * - 支持配置开关控制，保持向后兼容
 */

#ifndef ADAPTIVE_VOXEL_MANAGER_H
#define ADAPTIVE_VOXEL_MANAGER_H

#include <torch/torch.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>
#include <functional>
#include <Eigen/Eigen>

// 体素坐标结构
struct VoxelCoord {
    int x, y, z;
    int level;  // 分辨率级别 (0=最粗, higher=更细)
    
    VoxelCoord(int x_, int y_, int z_, int level_) : x(x_), y(y_), z(z_), level(level_) {}
    
    bool operator==(const VoxelCoord& other) const {
        return x == other.x && y == other.y && z == other.z && level == other.level;
    }
};

// 体素坐标哈希函数
struct VoxelCoordHash {
    std::size_t operator()(const VoxelCoord& coord) const {
        return std::hash<int>()(coord.x) ^ 
               (std::hash<int>()(coord.y) << 1) ^ 
               (std::hash<int>()(coord.z) << 2) ^
               (std::hash<int>()(coord.level) << 3);
    }
};

// 体素信息
struct VoxelInfo {
    std::vector<int> gaussian_indices;  // 该体素中的 Gaussian 索引
    float complexity_score;             // 几何复杂度评分
    float target_resolution;            // 目标分辨率
    int current_level;                  // 当前分辨率级别
    bool needs_refinement;              // 是否需要细化
    bool needs_coarsening;              // 是否需要粗化
    
    VoxelInfo() : complexity_score(0.0f), target_resolution(1.0f), 
                  current_level(0), needs_refinement(false), needs_coarsening(false) {}
};

// 几何复杂度计算器
class GeometryComplexityCalculator {
public:
    // 计算基于梯度的复杂度
    static float calculateGradientComplexity(
        const torch::Tensor& positions,
        const torch::Tensor& gradients,
        const std::vector<int>& indices,
        float radius = 0.1f
    );
    
    // 计算基于曲率的复杂度
    static float calculateCurvatureComplexity(
        const torch::Tensor& positions,
        const torch::Tensor& normals,
        const std::vector<int>& indices,
        float radius = 0.1f
    );
    
    // 计算基于密度变化的复杂度
    static float calculateDensityComplexity(
        const torch::Tensor& positions,
        const std::vector<int>& indices,
        float radius = 0.1f
    );
    
    // 综合复杂度评分
    static float calculateOverallComplexity(
        const torch::Tensor& positions,
        const torch::Tensor& gradients,
        const torch::Tensor& normals,
        const std::vector<int>& indices,
        float gradient_weight = 0.6f,      // 🔧 FIX: 增加梯度权重
        float curvature_weight = 0.1f,     // 🔧 FIX: 大幅降低曲率权重
        float density_weight = 0.3f        // 🔧 FIX: 增加密度权重
    );
};

// 自适应体素管理器
class AdaptiveVoxelManager {
public:
    // 构造函数
    AdaptiveVoxelManager(
        float base_voxel_size = 0.1f,
        int max_levels = 4,
        float complexity_threshold_refine = 0.7f,
        float complexity_threshold_coarsen = 0.3f,
        bool enabled = true
    );
    
    // 析构函数
    ~AdaptiveVoxelManager() = default;
    
    // 初始化体素网格
    void initialize(
        const torch::Tensor& initial_positions,
        const torch::Tensor& scene_bounds
    );
    
    // 更新体素网格（主要接口）
    void updateVoxelGrid(
        const torch::Tensor& positions,
        const torch::Tensor& gradients,
        const torch::Tensor& normals,
        const std::vector<int>& active_indices
    );
    
    // 获取需要密化的区域
    std::vector<int> getRegionsForDensification() const;
    
    // 获取需要剪枝的区域
    std::vector<int> getRegionsForPruning() const;
    
    // 获取推荐的采样密度
    std::vector<float> getRecommendedSamplingDensity(
        const torch::Tensor& positions
    ) const;
    
    // 检查是否应该在指定位置添加 Gaussian
    bool shouldAddGaussianAt(const torch::Tensor& position) const;
    
    // 检查是否应该移除指定的 Gaussian
    bool shouldRemoveGaussian(int gaussian_index, float opacity_threshold = 0.01f) const;
    
    // 获取指定位置的局部密度
    float getLocalDensity(const Eigen::Matrix<float, 3, 1>& position) const;
    
    // 获取指定位置的局部复杂度
    float getLocalComplexity(const Eigen::Matrix<float, 3, 1>& position) const;
    
    // 检查指定位置附近是否有高斯点
    bool hasNearbyGaussians(const Eigen::Matrix<float, 3, 1>& position, float radius) const;
    
    // 获取指定点的复杂度分数
    float getComplexityScore(int point_index) const;
    
    // 获取体素统计信息
    struct VoxelStats {
        int total_voxels;
        int active_voxels;
        std::vector<int> voxels_per_level;
        float average_complexity;
        float memory_usage_mb;
    };
    VoxelStats getStatistics() const;
    
    // 配置接口
    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_; }
    
    void setComplexityThresholds(float refine_threshold, float coarsen_threshold) {
        complexity_threshold_refine_ = refine_threshold;
        complexity_threshold_coarsen_ = coarsen_threshold;
    }
    
    void setMaxLevels(int max_levels) { max_levels_ = max_levels; }
    
    // 调试和可视化
    void exportVoxelGrid(const std::string& filename) const;
    std::vector<std::tuple<torch::Tensor, int, float>> getVoxelVisualizationData() const;

private:
    // 配置参数
    bool enabled_;
    float base_voxel_size_;
    int max_levels_;
    float complexity_threshold_refine_;
    float complexity_threshold_coarsen_;
    
    // 体素网格数据
    std::unordered_map<VoxelCoord, VoxelInfo, VoxelCoordHash> voxel_grid_;
    torch::Tensor scene_bounds_;  // [min_x, min_y, min_z, max_x, max_y, max_z]
    
    // 缓存和优化
    mutable std::unordered_map<VoxelCoord, float, VoxelCoordHash> complexity_cache_;
    int update_counter_;
    int cache_update_interval_;
    
    // 内部方法
    VoxelCoord positionToVoxelCoord(const torch::Tensor& position, int level) const;
    torch::Tensor voxelCoordToPosition(const VoxelCoord& coord) const;
    float getVoxelSizeAtLevel(int level) const;
    
    void updateVoxelComplexity(
        const VoxelCoord& coord,
        const torch::Tensor& positions,
        const torch::Tensor& gradients,
        const torch::Tensor& normals
    );
    
    void refineVoxel(const VoxelCoord& coord);
    void coarsenVoxel(const VoxelCoord& coord);
    
    std::vector<VoxelCoord> getNeighborVoxels(const VoxelCoord& coord, int radius = 1) const;
    
    void redistributeGaussians(
        const VoxelCoord& old_coord,
        const std::vector<VoxelCoord>& new_coords
    );
    
    void clearCache();
    bool isValidVoxelCoord(const VoxelCoord& coord) const;
    
    // 性能优化
    void optimizeVoxelGrid();
    void removeEmptyVoxels();
    void mergeCompatibleVoxels();
};

// 与 CoarseToFineManager 的集成接口
class VoxelCoarseToFineIntegration {
public:
    static void integrateWithCoarseToFine(
        AdaptiveVoxelManager& voxel_manager,
        class CoarseToFineManager& coarse_to_fine_manager
    );
    
    // 基于体素复杂度调整密化策略
    static std::vector<int> filterDensificationCandidates(
        const std::vector<int>& candidates,
        const AdaptiveVoxelManager& voxel_manager,
        const torch::Tensor& positions
    );
    
    // 基于体素复杂度调整剪枝策略
    static std::vector<int> filterPruningCandidates(
        const std::vector<int>& candidates,
        const AdaptiveVoxelManager& voxel_manager,
        const torch::Tensor& positions
    );
    
    // 获取自适应采样权重
    static torch::Tensor getAdaptiveSamplingWeights(
        const torch::Tensor& positions,
        const AdaptiveVoxelManager& voxel_manager
    );
};

#endif // ADAPTIVE_VOXEL_MANAGER_H