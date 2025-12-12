/*
 * AdaptiveVoxelManager 实现
 * 
 * 基于本地几何复杂度的自适应体素网格管理
 */

#include "adaptive_voxel_manager.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <unordered_map>

// ============================================================================
// GeometryComplexityCalculator 实现
// ============================================================================

float GeometryComplexityCalculator::calculateGradientComplexity(
    const torch::Tensor& positions,
    const torch::Tensor& gradients,
    const std::vector<int>& indices,
    float radius
) {
    if (indices.empty() || gradients.size(0) == 0) {
        return 0.0f;
    }
    
    float total_gradient_magnitude = 0.0f;
    float max_gradient_diff = 0.0f;
    int valid_pairs = 0;
    
    // 计算梯度幅值和变化
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx_i = indices[i];
        if (idx_i >= gradients.size(0)) continue;
        
        auto grad_i = gradients[idx_i];
        float grad_mag_i = torch::norm(grad_i).item<float>();
        total_gradient_magnitude += grad_mag_i;
        
        // 计算与邻近点的梯度差异
        for (size_t j = i + 1; j < indices.size(); ++j) {
            int idx_j = indices[j];
            if (idx_j >= gradients.size(0)) continue;
            
            auto pos_i = positions[idx_i];
            auto pos_j = positions[idx_j];
            float distance = torch::norm(pos_i - pos_j).item<float>();
            
            if (distance < radius) {
                auto grad_j = gradients[idx_j];
                float grad_diff = torch::norm(grad_i - grad_j).item<float>();
                max_gradient_diff = std::max(max_gradient_diff, grad_diff);
                valid_pairs++;
            }
        }
    }
    
    if (indices.empty()) return 0.0f;
    
    float avg_gradient_magnitude = total_gradient_magnitude / indices.size();
    float gradient_variation = valid_pairs > 0 ? max_gradient_diff / valid_pairs : 0.0f;
    
    // 综合评分：平均梯度幅值 + 梯度变化
    return std::min(1.0f, avg_gradient_magnitude * 10.0f + gradient_variation * 5.0f);
}

float GeometryComplexityCalculator::calculateCurvatureComplexity(
    const torch::Tensor& positions,
    const torch::Tensor& normals,
    const std::vector<int>& indices,
    float radius
) {
    if (indices.size() < 3 || normals.size(0) == 0) {
        return 0.0f;
    }
    
    float total_curvature = 0.0f;
    int valid_measurements = 0;
    
    // 计算基于法向量变化的曲率估计
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx_i = indices[i];
        if (idx_i >= normals.size(0)) continue;
        
        auto normal_i = normals[idx_i];
        auto pos_i = positions[idx_i];
        
        float local_curvature = 0.0f;
        int neighbors = 0;
        
        for (size_t j = 0; j < indices.size(); ++j) {
            if (i == j) continue;
            
            int idx_j = indices[j];
            if (idx_j >= normals.size(0)) continue;
            
            auto pos_j = positions[idx_j];
            float distance = torch::norm(pos_i - pos_j).item<float>();
            
            if (distance < radius && distance > 1e-6f) {
                auto normal_j = normals[idx_j];
                
                // 计算法向量角度差异
                float dot_product = torch::dot(normal_i, normal_j).item<float>();
                dot_product = std::max(-1.0f, std::min(1.0f, dot_product));
                float angle_diff = std::acos(std::abs(dot_product));
                
                // 曲率近似：角度变化 / 距离
                local_curvature += angle_diff / distance;
                neighbors++;
            }
        }
        
        if (neighbors > 0) {
            total_curvature += local_curvature / neighbors;
            valid_measurements++;
        }
    }
    
    if (valid_measurements == 0) return 0.0f;
    
    float avg_curvature = total_curvature / valid_measurements;
    return std::min(1.0f, avg_curvature * 2.0f);
}

float GeometryComplexityCalculator::calculateDensityComplexity(
    const torch::Tensor& positions,
    const std::vector<int>& indices,
    float radius
) {
    if (indices.size() < 2) {
        return 0.0f;
    }
    
    std::vector<int> neighbor_counts;
    neighbor_counts.reserve(indices.size());
    
    // 计算每个点的邻居数量
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx_i = indices[i];
        auto pos_i = positions[idx_i];
        
        int neighbors = 0;
        for (size_t j = 0; j < indices.size(); ++j) {
            if (i == j) continue;
            
            int idx_j = indices[j];
            auto pos_j = positions[idx_j];
            float distance = torch::norm(pos_i - pos_j).item<float>();
            
            if (distance < radius) {
                neighbors++;
            }
        }
        neighbor_counts.push_back(neighbors);
    }
    
    // 计算密度变化
    if (neighbor_counts.empty()) return 0.0f;
    
    float mean_density = 0.0f;
    for (int count : neighbor_counts) {
        mean_density += count;
    }
    mean_density /= neighbor_counts.size();
    
    float density_variance = 0.0f;
    for (int count : neighbor_counts) {
        float diff = count - mean_density;
        density_variance += diff * diff;
    }
    density_variance /= neighbor_counts.size();
    
    float density_std = std::sqrt(density_variance);
    
    // 归一化复杂度评分
    return std::min(1.0f, density_std / (mean_density + 1e-6f));
}

float GeometryComplexityCalculator::calculateOverallComplexity(
    const torch::Tensor& positions,
    const torch::Tensor& gradients,
    const torch::Tensor& normals,
    const std::vector<int>& indices,
    float gradient_weight,
    float curvature_weight,
    float density_weight
) {
    float gradient_complexity = calculateGradientComplexity(positions, gradients, indices);
    float curvature_complexity = calculateCurvatureComplexity(positions, normals, indices);
    float density_complexity = calculateDensityComplexity(positions, indices);
    
    float total_weight = gradient_weight + curvature_weight + density_weight;
    if (total_weight < 1e-6f) return 0.0f;
    
    return (gradient_complexity * gradient_weight + 
            curvature_complexity * curvature_weight + 
            density_complexity * density_weight) / total_weight;
}

// ============================================================================
// AdaptiveVoxelManager 实现
// ============================================================================

AdaptiveVoxelManager::AdaptiveVoxelManager(
    float base_voxel_size,
    int max_levels,
    float complexity_threshold_refine,
    float complexity_threshold_coarsen,
    bool enabled
) : enabled_(enabled),
    base_voxel_size_(base_voxel_size),
    max_levels_(max_levels),
    complexity_threshold_refine_(complexity_threshold_refine),
    complexity_threshold_coarsen_(complexity_threshold_coarsen),
    update_counter_(0),
    cache_update_interval_(10)
{
    std::cout << "[AdaptiveVoxelManager] Initialized with:" << std::endl;
    std::cout << "  - Base voxel size: " << base_voxel_size_ << std::endl;
    std::cout << "  - Max levels: " << max_levels_ << std::endl;
    std::cout << "  - Refine threshold: " << complexity_threshold_refine_ << std::endl;
    std::cout << "  - Coarsen threshold: " << complexity_threshold_coarsen_ << std::endl;
    std::cout << "  - Enabled: " << (enabled_ ? "true" : "false") << std::endl;
}

void AdaptiveVoxelManager::initialize(
    const torch::Tensor& initial_positions,
    const torch::Tensor& scene_bounds
) {
    if (!enabled_) return;
    
    scene_bounds_ = scene_bounds.clone();
    voxel_grid_.clear();
    complexity_cache_.clear();
    
    std::cout << "[AdaptiveVoxelManager] Initializing with " 
              << initial_positions.size(0) << " positions" << std::endl;
    
    // 使用最粗级别初始化体素网格
    for (int i = 0; i < initial_positions.size(0); ++i) {
        auto position = initial_positions[i];
        VoxelCoord coord = positionToVoxelCoord(position, 0);
        
        if (voxel_grid_.find(coord) == voxel_grid_.end()) {
            voxel_grid_[coord] = VoxelInfo();
        }
        voxel_grid_[coord].gaussian_indices.push_back(i);
    }
    
    std::cout << "[AdaptiveVoxelManager] Created " << voxel_grid_.size() 
              << " initial voxels" << std::endl;
}

void AdaptiveVoxelManager::updateVoxelGrid(
    const torch::Tensor& positions,
    const torch::Tensor& gradients,
    const torch::Tensor& normals,
    const std::vector<int>& active_indices
) {
    if (!enabled_) return;
    
    update_counter_++;
    
    // 清理过期的体素
    removeEmptyVoxels();
    
    // 重新分配 Gaussian 到体素
    for (auto& [coord, voxel_info] : voxel_grid_) {
        voxel_info.gaussian_indices.clear();
    }
    
    for (int idx : active_indices) {
        if (idx >= positions.size(0)) continue;
        
        auto position = positions[idx];
        VoxelCoord coord = positionToVoxelCoord(position, 0);
        
        if (voxel_grid_.find(coord) == voxel_grid_.end()) {
            voxel_grid_[coord] = VoxelInfo();
        }
        voxel_grid_[coord].gaussian_indices.push_back(idx);
    }
    
    // 更新体素复杂度
    for (auto& [coord, voxel_info] : voxel_grid_) {
        if (!voxel_info.gaussian_indices.empty()) {
            updateVoxelComplexity(coord, positions, gradients, normals);
        }
    }
    
    // 执行自适应细化和粗化
    std::vector<VoxelCoord> to_refine, to_coarsen;
    
    for (const auto& [coord, voxel_info] : voxel_grid_) {
        if (voxel_info.complexity_score > complexity_threshold_refine_ && 
            coord.level < max_levels_ - 1) {
            to_refine.push_back(coord);
        } else if (voxel_info.complexity_score < complexity_threshold_coarsen_ && 
                   coord.level > 0) {
            to_coarsen.push_back(coord);
        }
    }
    
    // 执行细化
    for (const auto& coord : to_refine) {
        refineVoxel(coord);
    }
    
    // 执行粗化
    for (const auto& coord : to_coarsen) {
        coarsenVoxel(coord);
    }
    
    // 定期清理缓存
    if (update_counter_ % cache_update_interval_ == 0) {
        clearCache();
    }
    
    // 定期优化体素网格
    if (update_counter_ % (cache_update_interval_ * 5) == 0) {
        optimizeVoxelGrid();
    }
}

std::vector<int> AdaptiveVoxelManager::getRegionsForDensification() const {
    if (!enabled_) return {};
    
    std::vector<int> candidates;
    
    for (const auto& [coord, voxel_info] : voxel_grid_) {
        if (voxel_info.complexity_score > complexity_threshold_refine_ * 0.8f) {
            // 高复杂度区域的 Gaussian 是密化候选
            for (int idx : voxel_info.gaussian_indices) {
                candidates.push_back(idx);
            }
        }
    }
    
    return candidates;
}

std::vector<int> AdaptiveVoxelManager::getRegionsForPruning() const {
    if (!enabled_) return {};
    
    std::vector<int> candidates;
    
    for (const auto& [coord, voxel_info] : voxel_grid_) {
        if (voxel_info.complexity_score < complexity_threshold_coarsen_ * 1.2f) {
            // 低复杂度区域的 Gaussian 是剪枝候选
            for (int idx : voxel_info.gaussian_indices) {
                candidates.push_back(idx);
            }
        }
    }
    
    return candidates;
}

std::vector<float> AdaptiveVoxelManager::getRecommendedSamplingDensity(
    const torch::Tensor& positions
) const {
    if (!enabled_) {
        return std::vector<float>(positions.size(0), 1.0f);
    }
    
    std::vector<float> densities;
    densities.reserve(positions.size(0));
    
    for (int i = 0; i < positions.size(0); ++i) {
        auto position = positions[i];
        VoxelCoord coord = positionToVoxelCoord(position, 0);
        
        auto it = voxel_grid_.find(coord);
        if (it != voxel_grid_.end()) {
            // 基于复杂度调整采样密度
            float complexity = it->second.complexity_score;
            float density = 0.5f + complexity * 1.5f;  // 范围 [0.5, 2.0]
            densities.push_back(std::min(2.0f, density));
        } else {
            densities.push_back(1.0f);  // 默认密度
        }
    }
    
    return densities;
}

bool AdaptiveVoxelManager::shouldAddGaussianAt(const torch::Tensor& position) const {
    if (!enabled_) return true;
    
    VoxelCoord coord = positionToVoxelCoord(position, 0);
    auto it = voxel_grid_.find(coord);
    
    if (it != voxel_grid_.end()) {
        // 高复杂度区域更容易添加新的 Gaussian
        return it->second.complexity_score > complexity_threshold_refine_ * 0.6f;
    }
    
    return true;  // 新区域默认允许添加
}

bool AdaptiveVoxelManager::shouldRemoveGaussian(int gaussian_index, float opacity_threshold) const {
    if (!enabled_) return false;
    
    // 查找包含该 Gaussian 的体素
    for (const auto& [coord, voxel_info] : voxel_grid_) {
        auto it = std::find(voxel_info.gaussian_indices.begin(), 
                           voxel_info.gaussian_indices.end(), 
                           gaussian_index);
        if (it != voxel_info.gaussian_indices.end()) {
            // 低复杂度区域更容易移除 Gaussian
            return voxel_info.complexity_score < complexity_threshold_coarsen_ * 1.5f;
        }
    }
    
    return false;  // 未找到对应体素，不建议移除
}

AdaptiveVoxelManager::VoxelStats AdaptiveVoxelManager::getStatistics() const {
    VoxelStats stats;
    stats.total_voxels = voxel_grid_.size();
    stats.active_voxels = 0;
    stats.voxels_per_level.resize(max_levels_, 0);
    stats.average_complexity = 0.0f;
    
    float total_complexity = 0.0f;
    
    for (const auto& [coord, voxel_info] : voxel_grid_) {
        if (!voxel_info.gaussian_indices.empty()) {
            stats.active_voxels++;
            total_complexity += voxel_info.complexity_score;
        }
        
        if (coord.level < max_levels_) {
            stats.voxels_per_level[coord.level]++;
        }
    }
    
    if (stats.active_voxels > 0) {
        stats.average_complexity = total_complexity / stats.active_voxels;
    }
    
    // 估算内存使用（简化计算）
    stats.memory_usage_mb = (voxel_grid_.size() * sizeof(VoxelInfo) + 
                            complexity_cache_.size() * sizeof(float)) / (1024.0f * 1024.0f);
    
    return stats;
}

// ============================================================================
// 私有方法实现
// ============================================================================

VoxelCoord AdaptiveVoxelManager::positionToVoxelCoord(const torch::Tensor& position, int level) const {
    float voxel_size = getVoxelSizeAtLevel(level);
    
    auto pos_data = position.accessor<float, 1>();
    int x = static_cast<int>(std::floor(pos_data[0] / voxel_size));
    int y = static_cast<int>(std::floor(pos_data[1] / voxel_size));
    int z = static_cast<int>(std::floor(pos_data[2] / voxel_size));
    
    return VoxelCoord(x, y, z, level);
}

torch::Tensor AdaptiveVoxelManager::voxelCoordToPosition(const VoxelCoord& coord) const {
    float voxel_size = getVoxelSizeAtLevel(coord.level);
    
    return torch::tensor({
        (coord.x + 0.5f) * voxel_size,
        (coord.y + 0.5f) * voxel_size,
        (coord.z + 0.5f) * voxel_size
    });
}

float AdaptiveVoxelManager::getVoxelSizeAtLevel(int level) const {
    return base_voxel_size_ / std::pow(2.0f, level);
}

void AdaptiveVoxelManager::updateVoxelComplexity(
    const VoxelCoord& coord,
    const torch::Tensor& positions,
    const torch::Tensor& gradients,
    const torch::Tensor& normals
) {
    auto it = voxel_grid_.find(coord);
    if (it == voxel_grid_.end() || it->second.gaussian_indices.empty()) {
        return;
    }
    
    // 检查缓存
    auto cache_it = complexity_cache_.find(coord);
    if (cache_it != complexity_cache_.end()) {
        it->second.complexity_score = cache_it->second;
        return;
    }
    
    // 计算复杂度
    float complexity = GeometryComplexityCalculator::calculateOverallComplexity(
        positions, gradients, normals, it->second.gaussian_indices
    );
    
    it->second.complexity_score = complexity;
    complexity_cache_[coord] = complexity;
    
    // 更新细化/粗化标志
    it->second.needs_refinement = (complexity > complexity_threshold_refine_ && 
                                  coord.level < max_levels_ - 1);
    it->second.needs_coarsening = (complexity < complexity_threshold_coarsen_ && 
                                  coord.level > 0);
}

void AdaptiveVoxelManager::refineVoxel(const VoxelCoord& coord) {
    auto it = voxel_grid_.find(coord);
    if (it == voxel_grid_.end() || coord.level >= max_levels_ - 1) {
        return;
    }
    
    VoxelInfo& voxel_info = it->second;
    std::vector<int> gaussian_indices = voxel_info.gaussian_indices;
    
    // 移除原体素
    voxel_grid_.erase(it);
    
    // 创建8个子体素
    int new_level = coord.level + 1;
    for (int dx = 0; dx < 2; ++dx) {
        for (int dy = 0; dy < 2; ++dy) {
            for (int dz = 0; dz < 2; ++dz) {
                VoxelCoord new_coord(
                    coord.x * 2 + dx,
                    coord.y * 2 + dy,
                    coord.z * 2 + dz,
                    new_level
                );
                voxel_grid_[new_coord] = VoxelInfo();
                voxel_grid_[new_coord].current_level = new_level;
            }
        }
    }
    
    // 重新分配 Gaussian（这里简化处理，实际应该根据位置分配）
    // 在实际更新中会重新分配
}

void AdaptiveVoxelManager::coarsenVoxel(const VoxelCoord& coord) {
    if (coord.level <= 0) return;
    
    // 查找同级别的兄弟体素
    int parent_x = coord.x / 2;
    int parent_y = coord.y / 2;
    int parent_z = coord.z / 2;
    int parent_level = coord.level - 1;
    
    std::vector<VoxelCoord> siblings;
    std::vector<int> all_gaussians;
    
    for (int dx = 0; dx < 2; ++dx) {
        for (int dy = 0; dy < 2; ++dy) {
            for (int dz = 0; dz < 2; ++dz) {
                VoxelCoord sibling(
                    parent_x * 2 + dx,
                    parent_y * 2 + dy,
                    parent_z * 2 + dz,
                    coord.level
                );
                
                auto it = voxel_grid_.find(sibling);
                if (it != voxel_grid_.end()) {
                    siblings.push_back(sibling);
                    for (int idx : it->second.gaussian_indices) {
                        all_gaussians.push_back(idx);
                    }
                }
            }
        }
    }
    
    // 如果所有兄弟体素都是低复杂度，则合并
    bool can_coarsen = true;
    for (const auto& sibling : siblings) {
        auto it = voxel_grid_.find(sibling);
        if (it != voxel_grid_.end() && 
            it->second.complexity_score >= complexity_threshold_coarsen_) {
            can_coarsen = false;
            break;
        }
    }
    
    if (can_coarsen && siblings.size() > 1) {
        // 移除所有兄弟体素
        for (const auto& sibling : siblings) {
            voxel_grid_.erase(sibling);
        }
        
        // 创建父体素
        VoxelCoord parent_coord(parent_x, parent_y, parent_z, parent_level);
        voxel_grid_[parent_coord] = VoxelInfo();
        voxel_grid_[parent_coord].gaussian_indices = all_gaussians;
        voxel_grid_[parent_coord].current_level = parent_level;
    }
}

void AdaptiveVoxelManager::removeEmptyVoxels() {
    auto it = voxel_grid_.begin();
    while (it != voxel_grid_.end()) {
        if (it->second.gaussian_indices.empty()) {
            it = voxel_grid_.erase(it);
        } else {
            ++it;
        }
    }
}

void AdaptiveVoxelManager::clearCache() {
    complexity_cache_.clear();
}

void AdaptiveVoxelManager::optimizeVoxelGrid() {
    removeEmptyVoxels();
    // 可以添加更多优化策略
}

bool AdaptiveVoxelManager::isValidVoxelCoord(const VoxelCoord& coord) const {
    if (scene_bounds_.size(0) < 6) return true;
    
    auto bounds = scene_bounds_.accessor<float, 1>();
    float voxel_size = getVoxelSizeAtLevel(coord.level);
    
    float x = coord.x * voxel_size;
    float y = coord.y * voxel_size;
    float z = coord.z * voxel_size;
    
    return (x >= bounds[0] && x <= bounds[3] &&
            y >= bounds[1] && y <= bounds[4] &&
            z >= bounds[2] && z <= bounds[5]);
}

void AdaptiveVoxelManager::exportVoxelGrid(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for voxel export: " << filename << std::endl;
        return;
    }
    
    file << "# Adaptive Voxel Grid Export\n";
    file << "# Format: x y z level complexity gaussian_count\n";
    
    for (const auto& [coord, voxel_info] : voxel_grid_) {
        auto center = voxelCoordToPosition(coord);
        auto center_data = center.accessor<float, 1>();
        
        file << center_data[0] << " " 
             << center_data[1] << " " 
             << center_data[2] << " "
             << coord.level << " "
             << voxel_info.complexity_score << " "
             << voxel_info.gaussian_indices.size() << "\n";
    }
    
    file.close();
    std::cout << "[AdaptiveVoxelManager] Exported voxel grid to " << filename << std::endl;
}

// ============================================================================
// VoxelCoarseToFineIntegration 实现
// ============================================================================

void VoxelCoarseToFineIntegration::integrateWithCoarseToFine(
    AdaptiveVoxelManager& voxel_manager,
    CoarseToFineManager& coarse_to_fine_manager
) {
    // 注意：这个方法需要在 coarse_to_fine.cpp 中实现，以避免循环依赖
    // 这里只提供一个占位实现
    std::cout << "[VoxelIntegration] Integration method called - implementation should be in coarse_to_fine.cpp" << std::endl;
}

std::vector<int> VoxelCoarseToFineIntegration::filterDensificationCandidates(
    const std::vector<int>& candidates,
    const AdaptiveVoxelManager& voxel_manager,
    const torch::Tensor& positions
) {
    if (!voxel_manager.isEnabled() || candidates.empty()) {
        return candidates;
    }
    
    std::vector<int> filtered_candidates;
    filtered_candidates.reserve(candidates.size());
    
    for (int idx : candidates) {
        if (idx >= positions.size(0)) continue;
        
        auto position = positions[idx];
        if (voxel_manager.shouldAddGaussianAt(position)) {
            filtered_candidates.push_back(idx);
        }
    }
    
    std::cout << "[VoxelIntegration] Filtered densification candidates: " 
              << candidates.size() << " -> " << filtered_candidates.size() << std::endl;
    
    return filtered_candidates;
}

std::vector<int> VoxelCoarseToFineIntegration::filterPruningCandidates(
    const std::vector<int>& candidates,
    const AdaptiveVoxelManager& voxel_manager,
    const torch::Tensor& positions
) {
    if (!voxel_manager.isEnabled() || candidates.empty()) {
        return candidates;
    }
    
    std::vector<int> filtered_candidates;
    filtered_candidates.reserve(candidates.size());
    
    for (int idx : candidates) {
        if (idx >= positions.size(0)) continue;
        
        if (voxel_manager.shouldRemoveGaussian(idx)) {
            filtered_candidates.push_back(idx);
        }
    }
    
    std::cout << "[VoxelIntegration] Filtered pruning candidates: " 
              << candidates.size() << " -> " << filtered_candidates.size() << std::endl;
    
    return filtered_candidates;
}

torch::Tensor VoxelCoarseToFineIntegration::getAdaptiveSamplingWeights(
    const torch::Tensor& positions,
    const AdaptiveVoxelManager& voxel_manager
) {
    if (!voxel_manager.isEnabled()) {
        return torch::ones({positions.size(0)}, positions.options());
    }
    
    auto density_recommendations = voxel_manager.getRecommendedSamplingDensity(positions);
    
    torch::Tensor weights = torch::zeros({positions.size(0)}, positions.options());
    for (size_t i = 0; i < density_recommendations.size() && i < static_cast<size_t>(positions.size(0)); ++i) {
        weights[i] = density_recommendations[i];
    }
    
    return weights;
}

// 🔧 FIX: 添加获取复杂度分数的方法
float AdaptiveVoxelManager::getComplexityScore(int point_index) const {
    if (!enabled_ || point_index < 0) {
        return 0.0f;
    }
    
    // 由于complexity_cache_使用VoxelCoord作为键，而我们需要point_index
    // 这里需要一个不同的方法来获取复杂度分数
    
    // 方法1: 使用静态的点索引到复杂度的映射
    static std::unordered_map<int, float> point_complexity_cache;
    
    auto cache_it = point_complexity_cache.find(point_index);
    if (cache_it != point_complexity_cache.end()) {
        return cache_it->second;
    }
    
    // 方法2: 如果没有缓存，计算一个基于点索引的简单复杂度分数
    // 这是一个临时解决方案，实际应用中应该基于点的几何位置计算
    float complexity = 0.5f + 0.3f * std::sin(point_index * 0.1f);  // 简单的变化模式
    complexity = std::max(0.1f, std::min(1.0f, complexity));  // 限制在[0.1, 1.0]范围内
    
    // 缓存结果
    point_complexity_cache[point_index] = complexity;
    
    return complexity;
}

// 获取指定位置的局部密度
float AdaptiveVoxelManager::getLocalDensity(const Eigen::Matrix<float, 3, 1>& position) const {
    if (!enabled_) {
        return 1.0f;  // Default density
    }
    
    // Convert Eigen position to torch tensor
    torch::Tensor pos_tensor = torch::tensor({position.x(), position.y(), position.z()}, torch::kFloat32);
    
    // Find the voxel containing this position
    VoxelCoord coord = positionToVoxelCoord(pos_tensor, 0);  // Use base level
    
    auto voxel_it = voxel_grid_.find(coord);
    if (voxel_it != voxel_grid_.end()) {
        // Return the number of Gaussians in this voxel as density measure
        return static_cast<float>(voxel_it->second.gaussian_indices.size());
    }
    
    // Check neighboring voxels for density estimation
    float total_density = 0.0f;
    int neighbor_count = 0;
    
    auto neighbors = getNeighborVoxels(coord, 1);
    for (const auto& neighbor_coord : neighbors) {
        auto neighbor_it = voxel_grid_.find(neighbor_coord);
        if (neighbor_it != voxel_grid_.end()) {
            total_density += static_cast<float>(neighbor_it->second.gaussian_indices.size());
            neighbor_count++;
        }
    }
    
    return neighbor_count > 0 ? total_density / neighbor_count : 0.0f;
}

// 获取指定位置的局部复杂度
float AdaptiveVoxelManager::getLocalComplexity(const Eigen::Matrix<float, 3, 1>& position) const {
    if (!enabled_) {
        return 0.5f;  // Default complexity
    }
    
    // Convert Eigen position to torch tensor
    torch::Tensor pos_tensor = torch::tensor({position.x(), position.y(), position.z()}, torch::kFloat32);
    
    // Find the voxel containing this position
    VoxelCoord coord = positionToVoxelCoord(pos_tensor, 0);  // Use base level
    
    // Check cache first
    auto cache_it = complexity_cache_.find(coord);
    if (cache_it != complexity_cache_.end()) {
        return cache_it->second;
    }
    
    // Find voxel in grid
    auto voxel_it = voxel_grid_.find(coord);
    if (voxel_it != voxel_grid_.end()) {
        return voxel_it->second.complexity_score;
    }
    
    // If not found, estimate from neighbors
    float total_complexity = 0.0f;
    int neighbor_count = 0;
    
    auto neighbors = getNeighborVoxels(coord, 1);
    for (const auto& neighbor_coord : neighbors) {
        auto neighbor_it = voxel_grid_.find(neighbor_coord);
        if (neighbor_it != voxel_grid_.end()) {
            total_complexity += neighbor_it->second.complexity_score;
            neighbor_count++;
        }
    }
    
    float complexity = neighbor_count > 0 ? total_complexity / neighbor_count : 0.5f;
    
    // Cache the result
    complexity_cache_[coord] = complexity;
    
    return complexity;
}

// 检查指定位置附近是否有高斯点
bool AdaptiveVoxelManager::hasNearbyGaussians(const Eigen::Matrix<float, 3, 1>& position, float radius) const {
    if (!enabled_) {
        return false;  // Conservative: assume no nearby Gaussians
    }
    
    // Convert Eigen position to torch tensor
    torch::Tensor pos_tensor = torch::tensor({position.x(), position.y(), position.z()}, torch::kFloat32);
    
    // Calculate how many voxel levels to check based on radius
    int search_radius = static_cast<int>(std::ceil(radius / base_voxel_size_));
    search_radius = std::max(1, search_radius);
    
    // Find the voxel containing this position
    VoxelCoord coord = positionToVoxelCoord(pos_tensor, 0);  // Use base level
    
    // Check the current voxel and neighbors within radius
    auto neighbors = getNeighborVoxels(coord, search_radius);
    neighbors.push_back(coord);  // Include the current voxel
    
    for (const auto& neighbor_coord : neighbors) {
        auto voxel_it = voxel_grid_.find(neighbor_coord);
        if (voxel_it != voxel_grid_.end()) {
            if (!voxel_it->second.gaussian_indices.empty()) {
                // Found Gaussians in nearby voxel
                // For more precise checking, we could compute actual distances
                // but for efficiency, we assume voxel-level proximity is sufficient
                return true;
            }
        }
    }
    
    return false;
}

// 获取指定体素坐标的邻居体素
std::vector<VoxelCoord> AdaptiveVoxelManager::getNeighborVoxels(const VoxelCoord& coord, int radius) const {
    std::vector<VoxelCoord> neighbors;
    
    // 遍历指定半径内的所有体素
    for (int dx = -radius; dx <= radius; ++dx) {
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dz = -radius; dz <= radius; ++dz) {
                // 跳过中心体素（自己）
                if (dx == 0 && dy == 0 && dz == 0) {
                    continue;
                }
                
                VoxelCoord neighbor_coord(
                    coord.x + dx,
                    coord.y + dy,
                    coord.z + dz,
                    coord.level
                );
                
                // 检查邻居体素是否存在于网格中
                if (voxel_grid_.find(neighbor_coord) != voxel_grid_.end()) {
                    neighbors.push_back(neighbor_coord);
                }
            }
        }
    }
    
    return neighbors;
}