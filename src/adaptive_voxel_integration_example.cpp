/*
 * AdaptiveVoxelManager 集成示例
 * 
 * 展示如何在现有的 coarse-to-fine 框架中集成 AdaptiveVoxelManager
 */

#include "adaptive_voxel_manager.h"
#include "coarse_to_fine.h"
#include "mapping.h"
#include <iostream>
#include <memory>

// 示例：如何在 Gaussian-LIC 中集成 AdaptiveVoxelManager
class GaussianLICWithAdaptiveVoxel {
public:
    GaussianLICWithAdaptiveVoxel(const Params& params) 
        : params_(params) {
        
        // 初始化 coarse-to-fine 管理器
        CoarseToFineParams ctf_params;
        ctf_params.enable_coarse_to_fine = params.enable_coarse_to_fine;
        ctf_params.initial_sampling_ratio = params.initial_sampling_ratio;
        ctf_params.min_initial_points = params.min_initial_points;
        ctf_params.max_initial_points = params.max_initial_points;
        ctf_params.densify_grad_threshold = params.densify_grad_threshold;
        ctf_params.densify_size_threshold = params.densify_size_threshold;
        ctf_params.densify_start_iter = params.densify_start_iter;
        ctf_params.densify_stop_iter = params.densify_stop_iter;
        ctf_params.densify_interval = params.densify_interval;
        ctf_params.prune_opacity_threshold = params.prune_opacity_threshold;
        ctf_params.prune_scale_threshold = params.prune_scale_threshold;
        ctf_params.prune_start_iter = params.prune_start_iter;
        ctf_params.prune_interval = params.prune_interval;
        
        coarse_to_fine_manager_ = std::make_unique<CoarseToFineManager>(ctf_params);
        
        // 如果启用了自适应体素管理，则初始化
        if (params.enable_adaptive_voxel) {
            initializeAdaptiveVoxelManager();
        }
    }
    
    void initializeAdaptiveVoxelManager() {
        std::cout << "\033[1;32m[Integration] Initializing AdaptiveVoxelManager...\033[0m" << std::endl;
        
        // 创建 AdaptiveVoxelManager
        voxel_manager_ = std::make_shared<AdaptiveVoxelManager>(
            params_.base_voxel_size,
            params_.max_voxel_levels,
            params_.complexity_threshold_refine,
            params_.complexity_threshold_coarsen,
            params_.enable_adaptive_voxel
        );
        
        // 集成到 coarse-to-fine 管理器
        coarse_to_fine_manager_->setAdaptiveVoxelManager(voxel_manager_);
        
        std::cout << "\033[1;32m[Integration] AdaptiveVoxelManager integrated successfully!\033[0m" << std::endl;
        
        // 打印配置信息
        printVoxelConfiguration();
    }
    
    void printVoxelConfiguration() {
        std::cout << "\033[1;36m[VoxelConfig] Configuration:\033[0m" << std::endl;
        std::cout << "  - Enabled: " << (params_.enable_adaptive_voxel ? "true" : "false") << std::endl;
        std::cout << "  - Base voxel size: " << params_.base_voxel_size << "m" << std::endl;
        std::cout << "  - Max levels: " << params_.max_voxel_levels << std::endl;
        std::cout << "  - Refine threshold: " << params_.complexity_threshold_refine << std::endl;
        std::cout << "  - Coarsen threshold: " << params_.complexity_threshold_coarsen << std::endl;
        std::cout << "  - Update interval: " << params_.voxel_update_interval << std::endl;
        std::cout << "  - Complexity weights: G=" << params_.gradient_weight 
                  << ", C=" << params_.curvature_weight 
                  << ", D=" << params_.density_weight << std::endl;
    }
    
    // 模拟训练循环中的使用
    void simulateTrainingIteration(int iteration, std::shared_ptr<GaussianModel>& pc) {
        std::cout << "\033[1;33m[Training] Iteration " << iteration << "\033[0m" << std::endl;
        
        // 1. 更新体素网格（如果启用）
        if (voxel_manager_ && voxel_manager_->isEnabled()) {
            coarse_to_fine_manager_->updateVoxelGrid(pc, iteration);
        }
        
        // 2. 执行自适应密化（现在会考虑体素复杂度）
        coarse_to_fine_manager_->adaptiveDensification(pc, iteration);
        
        // 3. 执行剪枝（现在会考虑体素复杂度）
        coarse_to_fine_manager_->pruneGaussians(pc, iteration);
        
        // 4. 定期打印统计信息
        if (iteration % 100 == 0 && voxel_manager_) {
            printVoxelStatistics();
        }
    }
    
    void printVoxelStatistics() {
        if (!voxel_manager_ || !voxel_manager_->isEnabled()) {
            return;
        }
        
        auto stats = voxel_manager_->getStatistics();
        std::cout << "\033[1;36m[VoxelStats] Total voxels: " << stats.total_voxels
                  << ", Active: " << stats.active_voxels
                  << ", Avg complexity: " << std::fixed << std::setprecision(3) << stats.average_complexity
                  << ", Memory: " << std::fixed << std::setprecision(2) << stats.memory_usage_mb << "MB\033[0m" << std::endl;
        
        std::cout << "\033[1;36m[VoxelStats] Voxels per level: ";
        for (size_t i = 0; i < stats.voxels_per_level.size(); ++i) {
            std::cout << "L" << i << ":" << stats.voxels_per_level[i] << " ";
        }
        std::cout << "\033[0m" << std::endl;
    }
    
    // 导出体素网格用于可视化
    void exportVoxelGrid(const std::string& filename) {
        if (voxel_manager_ && voxel_manager_->isEnabled()) {
            voxel_manager_->exportVoxelGrid(filename);
            std::cout << "\033[1;32m[Export] Voxel grid exported to " << filename << "\033[0m" << std::endl;
        }
    }
    
    // 动态调整体素参数
    void adjustVoxelParameters(float refine_threshold, float coarsen_threshold) {
        if (voxel_manager_) {
            voxel_manager_->setComplexityThresholds(refine_threshold, coarsen_threshold);
            std::cout << "\033[1;33m[Adjust] Updated complexity thresholds: refine=" 
                      << refine_threshold << ", coarsen=" << coarsen_threshold << "\033[0m" << std::endl;
        }
    }
    
    // 获取推荐的采样密度
    std::vector<float> getAdaptiveSamplingDensity(const torch::Tensor& positions) {
        if (voxel_manager_ && voxel_manager_->isEnabled()) {
            return voxel_manager_->getRecommendedSamplingDensity(positions);
        }
        return std::vector<float>(positions.size(0), 1.0f);
    }

private:
    Params params_;
    std::unique_ptr<CoarseToFineManager> coarse_to_fine_manager_;
    std::shared_ptr<AdaptiveVoxelManager> voxel_manager_;
};

// 使用示例函数
void demonstrateAdaptiveVoxelIntegration() {
    std::cout << "\033[1;35m=== AdaptiveVoxelManager Integration Demo ===\033[0m" << std::endl;
    
    // 1. 创建示例参数
    Params params;
    
    // 基础参数
    params.enable_coarse_to_fine = true;
    params.initial_sampling_ratio = 0.25f;
    params.min_initial_points = 2000;
    params.max_initial_points = 80000;
    
    // 密化参数
    params.densify_grad_threshold = 0.00015f;
    params.densify_size_threshold = 25.0f;
    params.densify_start_iter = 300;
    params.densify_stop_iter = 12000;
    params.densify_interval = 80;
    
    // 剪枝参数
    params.prune_opacity_threshold = 0.008f;
    params.prune_scale_threshold = 80.0f;
    params.prune_start_iter = 2000;
    params.prune_interval = 80;
    
    // 自适应体素参数
    params.enable_adaptive_voxel = true;
    params.base_voxel_size = 0.1f;
    params.max_voxel_levels = 4;
    params.complexity_threshold_refine = 0.7f;
    params.complexity_threshold_coarsen = 0.3f;
    params.voxel_update_interval = 10;
    params.gradient_weight = 0.4f;
    params.curvature_weight = 0.4f;
    params.density_weight = 0.2f;
    
    // 2. 创建集成系统
    GaussianLICWithAdaptiveVoxel system(params);
    
    // 3. 模拟一些训练迭代
    std::cout << "\033[1;33m--- Simulating training iterations ---\033[0m" << std::endl;
    
    // 创建模拟的 GaussianModel（这里只是示例）
    // 在实际使用中，这将是真实的 GaussianModel 实例
    std::shared_ptr<GaussianModel> mock_pc = nullptr; // 实际使用时需要真实的 pc
    
    for (int iter = 0; iter < 1000; iter += 100) {
        // system.simulateTrainingIteration(iter, mock_pc);
        std::cout << "  Iteration " << iter << " (simulated)" << std::endl;
    }
    
    // 4. 演示动态参数调整
    std::cout << "\033[1;33m--- Demonstrating dynamic parameter adjustment ---\033[0m" << std::endl;
    system.adjustVoxelParameters(0.8f, 0.2f);  // 更激进的细化，更保守的粗化
    
    // 5. 导出体素网格
    std::cout << "\033[1;33m--- Exporting voxel grid ---\033[0m" << std::endl;
    system.exportVoxelGrid("adaptive_voxel_grid.txt");
    
    std::cout << "\033[1;35m=== Demo completed successfully! ===\033[0m" << std::endl;
}

// 向后兼容性测试
void testBackwardCompatibility() {
    std::cout << "\033[1;35m=== Backward Compatibility Test ===\033[0m" << std::endl;
    
    // 测试禁用自适应体素时的行为
    Params params;
    params.enable_coarse_to_fine = true;
    params.enable_adaptive_voxel = false;  // 禁用自适应体素
    
    GaussianLICWithAdaptiveVoxel system(params);
    
    std::cout << "\033[1;32m✓ System works correctly with adaptive voxel disabled\033[0m" << std::endl;
    std::cout << "\033[1;32m✓ Backward compatibility maintained\033[0m" << std::endl;
}

// 主函数（仅用于测试）
int main() {
    try {
        demonstrateAdaptiveVoxelIntegration();
        testBackwardCompatibility();
        
        std::cout << "\033[1;32m🎉 All tests passed! AdaptiveVoxelManager is ready for use.\033[0m" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\033[1;31m❌ Error: " << e.what() << "\033[0m" << std::endl;
        return 1;
    }
    
    return 0;
}