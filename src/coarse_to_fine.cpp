/*
 * Gaussian-LIC: Real-Time Photo-Realistic SLAM with Gaussian Splatting and LiDAR-Inertial-Camera Fusion
 * Copyright (C) 2025 Xiaolei Lang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "coarse_to_fine.h"
#include "gaussian.h"
#include "general_utils.h"
#include "simple-knn/spatial.h"
#include "adaptive_voxel_manager.h"
#include "normal_estimation.h"  // 添加法向量估计
#include "rasterizer/renderer.h"  // 添加渲染器
#include "camera.h"  // 添加相机

#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>

// Global manager instance
std::unique_ptr<CoarseToFineManager> g_coarse_to_fine_manager = nullptr;

CoarseToFineManager::CoarseToFineManager(const CoarseToFineParams& params)
    : params_(params), total_iterations_(0),
      voxel_manager_(nullptr),
      num_split_operations_(0), num_clone_operations_(0), num_pruned_gaussians_(0)
{
    std::cout << "\033[1;36m[CoarseToFine] Initialized with SplaTAM-style processing\033[0m" << std::endl;
}

void CoarseToFineManager::initializeCoarse(std::shared_ptr<GaussianModel>& pc, 
                                          const std::shared_ptr<Dataset>& dataset)
{
    std::cout << "\033[1;36m[CoarseToFine] Starting SplaTAM-style initialization...\033[0m" << std::endl;
    
    int total_points = static_cast<int>(dataset->pointcloud_.size());
    if (total_points == 0) {
        std::cerr << "Error: No points in dataset for initialization!" << std::endl;
        return;
    }
    
    std::vector<int> sampled_indices;
    
    if (params_.use_all_points_initialization) {
        // SplaTAM style: use all valid points at once
        sampled_indices.resize(total_points);
        std::iota(sampled_indices.begin(), sampled_indices.end(), 0);
        
        // Optional: apply voxel-based spatial filtering for quality
        if (voxel_manager_ && voxel_manager_->isEnabled()) {
            std::vector<int> filtered_indices;
            for (int idx : sampled_indices) {
                torch::Tensor pos = torch::tensor({dataset->pointcloud_[idx].x(), 
                                                 dataset->pointcloud_[idx].y(), 
                                                 dataset->pointcloud_[idx].z()}, torch::kFloat32);
                if (shouldAddGaussianAt(pos)) {
                    filtered_indices.push_back(idx);
                }
            }
            sampled_indices = filtered_indices;
            std::cout << "\033[1;32m[CoarseToFine] Voxel filtering: " << filtered_indices.size() 
                      << " / " << total_points << " points retained\033[0m" << std::endl;
        }
    } else {
        // Fallback: use initial sampling ratio
        float initial_ratio = params_.initial_sampling_ratio;
        int target_points = static_cast<int>(total_points * initial_ratio);
        target_points = std::max(params_.min_initial_points, 
                               std::min(target_points, params_.max_initial_points));
        sampled_indices = samplePointIndices(total_points, 
                                           static_cast<float>(target_points) / total_points);
    }
    
    std::cout << "\033[1;36m[CoarseToFine] Using " << sampled_indices.size() 
              << " points from " << total_points << " total points\033[0m" << std::endl;
    
    // Create tensors for sampled points
    int num_sampled = sampled_indices.size();
    torch::Tensor fused_point_cloud = torch::zeros({num_sampled, 3}, torch::kFloat32).cuda();
    int deg_2 = (pc->sh_degree_ + 1) * (pc->sh_degree_ + 1);
    torch::Tensor features = torch::zeros({num_sampled, 3, deg_2}, torch::kFloat32).cuda();
    torch::Tensor scales = torch::zeros({num_sampled}, torch::kFloat32).cuda();
    
    double f = (dataset->fx_ + dataset->fy_) / 2;
    for (int i = 0; i < num_sampled; ++i) {
        int idx = sampled_indices[i];
        auto& pt_w = dataset->pointcloud_[idx];
        auto& color = dataset->pointcolor_[idx];
        
        fused_point_cloud.index({i, 0}) = pt_w.x();
        fused_point_cloud.index({i, 1}) = pt_w.y();
        fused_point_cloud.index({i, 2}) = pt_w.z();
        features.index({i, 0, 0}) = RGB2SH(color.x());
        features.index({i, 1, 0}) = RGB2SH(color.y());
        features.index({i, 2, 0}) = RGB2SH(color.z());
        
        double d = dataset->pointdepth_[idx];
        scales.index({i}) = std::log(pc->scaling_scale_ * d / f);
    }
    
    scales = scales.unsqueeze(1).repeat({1, 3});  // (n, 3)
    torch::Tensor rots = torch::zeros({num_sampled, 4}, torch::kFloat32).cuda();  // (n, 4)
    rots.index({torch::indexing::Slice(), 0}) = 1;
    torch::Tensor opacities = general_utils::inverse_sigmoid(0.1f * torch::ones({num_sampled, 1}, torch::kFloat32).cuda());
    
    // Handle skybox points if enabled
    if (pc->skybox_points_num_ > 0) {
        int sky_num = pc->skybox_points_num_;
        double radius = pc->skybox_radius_;
        torch::Tensor pi = torch::acos(torch::tensor(-1.0, torch::kFloat32).cuda());
        torch::Tensor theta = 2.0 * pi * torch::rand({sky_num}, torch::kFloat32).cuda();
        torch::Tensor phi = torch::acos(1.0 - 1.4 * torch::rand({sky_num}, torch::kFloat32).cuda());
        torch::Tensor sky_fused_point_cloud = torch::zeros({sky_num, 3}, torch::kFloat32).cuda();
        sky_fused_point_cloud.index({torch::indexing::Slice(), 0}) = radius * 10 * torch::cos(theta) * torch::sin(phi);
        sky_fused_point_cloud.index({torch::indexing::Slice(), 1}) = radius * 10 * torch::sin(theta) * torch::sin(phi);
        sky_fused_point_cloud.index({torch::indexing::Slice(), 2}) = radius * 10 * torch::cos(phi);
        
        torch::Tensor sky_features = torch::zeros({sky_num, 3, deg_2}, torch::kFloat32).cuda();
        sky_features.index({torch::indexing::Slice(), 0, 0}) = 0.7;
        sky_features.index({torch::indexing::Slice(), 1, 0}) = 0.8;
        sky_features.index({torch::indexing::Slice(), 2, 0}) = 0.95;
        
        torch::Tensor point_cloud_copy = sky_fused_point_cloud.clone();
        torch::Tensor dist2 = torch::clamp_min(distCUDA2(point_cloud_copy), 0.0000001);
        torch::Tensor sky_scales = torch::log(torch::sqrt(dist2));
        sky_scales = sky_scales.unsqueeze(1).repeat({1, 3});
        torch::Tensor sky_rots = torch::zeros({sky_num, 4}, torch::kFloat32).cuda();
        sky_rots.index({torch::indexing::Slice(), 0}) = 1;
        torch::Tensor sky_opacities = general_utils::inverse_sigmoid(0.7f * torch::ones({sky_num, 1}, torch::kFloat32).cuda());
        
        fused_point_cloud = torch::cat({sky_fused_point_cloud, fused_point_cloud}, 0);
        features = torch::cat({sky_features, features}, 0);
        scales = torch::cat({sky_scales, scales}, 0);
        rots = torch::cat({sky_rots, rots}, 0);
        opacities = torch::cat({sky_opacities, opacities}, 0);
    }
    
    // Set Gaussian model parameters
    pc->xyz_ = fused_point_cloud.requires_grad_();
    pc->features_dc_ = features.index({torch::indexing::Slice(),
                      torch::indexing::Slice(),
                      torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous().requires_grad_();
    pc->features_rest_ = features.index({torch::indexing::Slice(),
                      torch::indexing::Slice(),
                      torch::indexing::Slice(1, features.size(2))}).transpose(1, 2).contiguous().requires_grad_();
    pc->scaling_ = scales.requires_grad_();
    pc->rotation_ = rots.requires_grad_();
    pc->opacity_ = opacities.requires_grad_();
    
    if (pc->apply_exposure_) {
        torch::Tensor exposure = torch::eye(3, torch::kFloat32).cuda();
        exposure = torch::cat({exposure, torch::zeros({3, 1}, torch::kFloat32).cuda()}, 1);
        pc->exposure_ = exposure.requires_grad_();
    }
    
    // Update tensor vectors for optimizer
    pc->Tensor_vec_xyz_ = {pc->xyz_};
    pc->Tensor_vec_feature_dc_ = {pc->features_dc_};
    pc->Tensor_vec_feature_rest_ = {pc->features_rest_};
    pc->Tensor_vec_opacity_ = {pc->opacity_};
    pc->Tensor_vec_scaling_ = {pc->scaling_};
    pc->Tensor_vec_rotation_ = {pc->rotation_};
    pc->Tensor_vec_exposure_ = {pc->exposure_};
    
    std::cout << std::fixed << std::setprecision(2) 
              << "\033[1;36m[CoarseToFine] Initialized with " 
              << double(fused_point_cloud.size(0)) / 10000 << "w GS (coarse level)\033[0m" << std::endl;
}

void CoarseToFineManager::adaptiveDensification(std::shared_ptr<GaussianModel>& pc, int iteration)
{
    if (!shouldDensify(iteration)) {
        return;
    }
    
    // Update voxel grid first
    updateVoxelGrid(pc, iteration);
    
    // Get gradients from xyz parameter
    if (!pc->xyz_.grad().defined()) {
        return;
    }
    
    torch::Tensor grads = pc->xyz_.grad();
    torch::Tensor grad_norm = torch::norm(grads, 2, 1, true);  // L2 norm along dimension 1
    
    // Create gradient mask for high-gradient points
    torch::Tensor grad_mask = grad_norm.squeeze() > params_.densify_grad_threshold;
    
    // Get scaling to determine size
    torch::Tensor scales = pc->getScaling();
    torch::Tensor max_scale = std::get<0>(torch::max(scales, 1));
    
    // Create size mask for large vs small Gaussians
    torch::Tensor size_mask = max_scale > params_.densify_size_threshold;
    
    // Apply voxel-based filtering if available
    if (voxel_manager_ && voxel_manager_->isEnabled()) {
        // Get voxel-recommended densification regions
        auto voxel_candidates = voxel_manager_->getRegionsForDensification();
        
        if (!voxel_candidates.empty()) {
            // Create voxel mask
            torch::Tensor voxel_mask = torch::zeros_like(grad_mask);
            for (int idx : voxel_candidates) {
                if (idx < voxel_mask.size(0)) {
                    voxel_mask[idx] = true;
                }
            }
            
            // 🔧 FIX: 改为加法逻辑而不是保守的AND逻辑
            // 原逻辑：只有梯度大且体素复杂度高的区域才密化（保守）
            // 新逻辑：梯度大的区域 OR 体素复杂度高的区域都密化（积极）
            
            // 🔧 FIX: 简化的OR逻辑，避免复杂度查询问题
            // 直接使用体素推荐的候选点，而不是查询复杂度分数
            
            // 创建体素推荐的严格筛选
            // 只选择前50%的体素候选点，确保质量
            int max_voxel_additions = std::max(1, (int)(voxel_candidates.size() * 0.5));
            
            torch::Tensor enhanced_voxel_mask = torch::zeros_like(grad_mask);
            for (int i = 0; i < max_voxel_additions && i < (int)voxel_candidates.size(); ++i) {
                int idx = voxel_candidates[i];
                if (idx < enhanced_voxel_mask.size(0)) {
                    enhanced_voxel_mask[idx] = true;
                }
            }
            
            // 组合策略：原梯度区域 + 精选的体素区域
            grad_mask = grad_mask | enhanced_voxel_mask;
            
            std::cout << "\033[1;32m[CoarseToFine] Enhanced OR logic: " 
                      << "gradient candidates + " << max_voxel_additions 
                      << " voxel candidates\033[0m" << std::endl;
            
            std::cout << "\033[1;36m[CoarseToFine] Voxel-guided densification: " 
                      << voxel_candidates.size() << " candidates\033[0m" << std::endl;
        }
    }
    
    // Split large Gaussians with high gradients
    torch::Tensor split_mask = grad_mask & size_mask;
    if (split_mask.sum().item<int>() > 0) {
        splitGaussians(pc, split_mask, size_mask);
    }
    
    // Clone small Gaussians with high gradients
    torch::Tensor clone_mask = grad_mask & (~size_mask);
    if (clone_mask.sum().item<int>() > 0) {
        cloneGaussians(pc, clone_mask, size_mask);
    }
    
    if ((iteration % (params_.densify_interval * 10)) == 0) {
        std::cout << "\033[1;33m[CoarseToFine] Iter " << iteration 
                  << " - Split: " << num_split_operations_ 
                  << ", Clone: " << num_clone_operations_ 
                  << ", Total GS: " << pc->getXYZ().size(0) << "\033[0m" << std::endl;
    }
}

void CoarseToFineManager::pruneGaussians(std::shared_ptr<GaussianModel>& pc, int iteration)
{
    if (!shouldPrune(iteration)) {
        return;
    }
    
    torch::Tensor prune_mask = getPruneMask(pc);
    
    // Apply voxel-based filtering if available
    if (voxel_manager_ && voxel_manager_->isEnabled()) {
        // Get voxel-recommended pruning regions
        auto voxel_candidates = voxel_manager_->getRegionsForPruning();
        
        if (!voxel_candidates.empty()) {
            // Create voxel mask for pruning candidates
            torch::Tensor voxel_prune_mask = torch::zeros_like(prune_mask);
            for (int idx : voxel_candidates) {
                if (idx < voxel_prune_mask.size(0)) {
                    voxel_prune_mask[idx] = true;
                }
            }
            
            // Combine original prune mask with voxel recommendations
            // Only prune if both conditions are met
            prune_mask = prune_mask & voxel_prune_mask;
            
            std::cout << "\033[1;36m[CoarseToFine] Voxel-guided pruning: " 
                      << voxel_candidates.size() << " candidates\033[0m" << std::endl;
        }
    }
    
    int num_to_prune = prune_mask.sum().item<int>();
    
    if (num_to_prune > 0) {
        applyPruneMask(pc, prune_mask);
        num_pruned_gaussians_ += num_to_prune;
        
        if ((iteration % (params_.prune_interval * 10)) == 0) {
            std::cout << "\033[1;31m[CoarseToFine] Iter " << iteration 
                      << " - Pruned " << num_to_prune << " GS, Total: " 
                      << pc->getXYZ().size(0) << "\033[0m" << std::endl;
        }
    }
}

void CoarseToFineManager::addSilhouetteGaussians(std::shared_ptr<GaussianModel>& pc, 
                                                const std::shared_ptr<Camera>& viewpoint_camera,
                                                const torch::Tensor& gt_image,
                                                const torch::Tensor& gt_depth)
{
    // SplaTAM-style silhouette-guided densification
    std::cout << "\033[1;36m[CoarseToFine] Performing SplaTAM-style silhouette-guided densification...\033[0m" << std::endl;
    
    // TODO: Full implementation requires proper rendering integration
    // For now, this is a placeholder that demonstrates the concept
    
    // In a complete implementation, this function would:
    // 1. Render current view to get alpha/silhouette map
    // 2. Find holes where silhouette < 0.5 or depth error is large  
    // 3. Unproject pixels to 3D points
    // 4. Filter through voxel manager
    // 5. Add new Gaussians to the model
    
    std::cout << "\033[1;33m[CoarseToFine] SplaTAM silhouette densification placeholder called\033[0m" << std::endl;
    std::cout << "\033[1;33m[CoarseToFine] This would add Gaussians based on rendered alpha < 0.5\033[0m" << std::endl;
    
    // Example of how voxel manager integration would work:
    if (voxel_manager_ && voxel_manager_->isEnabled()) {
        std::cout << "\033[1;32m[CoarseToFine] Voxel manager is available for spatial filtering\033[0m" << std::endl;
    }
}

bool CoarseToFineManager::shouldAddGaussianAt(const torch::Tensor& position) const
{
    // Check with voxel manager if available
    if (voxel_manager_ && voxel_manager_->isEnabled()) {
        // Convert position to Eigen format for voxel manager
        Eigen::Matrix<float, 3, 1> pos_eigen;
        pos_eigen << position[0].item<float>(), 
                     position[1].item<float>(), 
                     position[2].item<float>();
        
        // Check if this position is in a region that allows new Gaussians
        // The voxel manager acts as a spatial filter to prevent over-densification
        
        // 1. Check if the region is not already over-densified
        float local_density = voxel_manager_->getLocalDensity(pos_eigen);
        if (local_density > 10.0f) {  // Too dense already
            return false;
        }
        
        // 2. Check if the region has reasonable complexity to warrant a new Gaussian
        float complexity = voxel_manager_->getLocalComplexity(pos_eigen);
        if (complexity < 0.1f) {  // Too simple, might not need new Gaussians
            return false;
        }
        
        // 3. Check spatial distance to existing Gaussians
        float min_distance = params_.spatial_sampling_threshold;
        if (voxel_manager_->hasNearbyGaussians(pos_eigen, min_distance)) {
            return false;  // Too close to existing Gaussians
        }
        
        return true;  // Passed all checks
    }
    
    // Default: allow adding points (fallback when voxel manager is not available)
    return true;
}

bool CoarseToFineManager::shouldDensify(int iteration) const
{
    return iteration >= params_.densify_start_iter && 
           iteration <= params_.densify_stop_iter &&
           (iteration % params_.densify_interval) == 0;
}

bool CoarseToFineManager::shouldPrune(int iteration) const
{
    return iteration >= params_.prune_start_iter &&
           (iteration % params_.prune_interval) == 0;
}

std::vector<int> CoarseToFineManager::samplePointIndices(int total_points, float sampling_ratio)
{
    int num_samples = static_cast<int>(total_points * sampling_ratio);
    num_samples = std::min(num_samples, total_points);
    
    std::vector<int> indices(total_points);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);
    
    indices.resize(num_samples);
    return indices;
}

void CoarseToFineManager::splitGaussians(std::shared_ptr<GaussianModel>& pc, 
                                        const torch::Tensor& grad_mask,
                                        const torch::Tensor& size_mask)
{
    // This is a simplified implementation
    // In practice, you would implement proper Gaussian splitting
    // by creating two new Gaussians from each split Gaussian
    
    torch::Tensor indices = torch::nonzero(grad_mask).squeeze(1);
    int num_to_split = indices.size(0);
    
    if (num_to_split == 0) return;
    
    // Get current parameters
    torch::Tensor xyz = pc->getXYZ();
    torch::Tensor features_dc = pc->getFeaturesDc();
    torch::Tensor features_rest = pc->getFeaturesRest();
    torch::Tensor scaling = pc->getScaling();
    torch::Tensor rotation = pc->getRotation();
    torch::Tensor opacity = pc->getOpacity();
    
    // Create new Gaussians by duplicating and slightly perturbing
    torch::Tensor new_xyz = xyz.index_select(0, indices);
    torch::Tensor new_features_dc = features_dc.index_select(0, indices);
    torch::Tensor new_features_rest = features_rest.index_select(0, indices);
    torch::Tensor new_scaling = scaling.index_select(0, indices) - std::log(1.6f);  // Reduce scale
    torch::Tensor new_rotation = rotation.index_select(0, indices);
    torch::Tensor new_opacity = opacity.index_select(0, indices);
    
    // Add small random perturbation to positions
    torch::Tensor perturbation = torch::randn_like(new_xyz) * 0.01f;
    new_xyz = new_xyz + perturbation;
    
    // Use densificationPostfix to add new Gaussians
    pc->densificationPostfix(new_xyz, new_features_dc, new_features_rest, 
                            new_opacity, new_scaling, new_rotation);
    
    num_split_operations_ += num_to_split;
}

void CoarseToFineManager::cloneGaussians(std::shared_ptr<GaussianModel>& pc,
                                        const torch::Tensor& grad_mask,
                                        const torch::Tensor& size_mask)
{
    torch::Tensor indices = torch::nonzero(grad_mask).squeeze(1);
    int num_to_clone = indices.size(0);
    
    if (num_to_clone == 0) return;
    
    // Get current parameters
    torch::Tensor xyz = pc->getXYZ();
    torch::Tensor features_dc = pc->getFeaturesDc();
    torch::Tensor features_rest = pc->getFeaturesRest();
    torch::Tensor scaling = pc->getScaling();
    torch::Tensor rotation = pc->getRotation();
    torch::Tensor opacity = pc->getOpacity();
    
    // Clone selected Gaussians
    torch::Tensor new_xyz = xyz.index_select(0, indices);
    torch::Tensor new_features_dc = features_dc.index_select(0, indices);
    torch::Tensor new_features_rest = features_rest.index_select(0, indices);
    torch::Tensor new_scaling = scaling.index_select(0, indices);
    torch::Tensor new_rotation = rotation.index_select(0, indices);
    torch::Tensor new_opacity = opacity.index_select(0, indices);
    
    // Use densificationPostfix to add new Gaussians
    pc->densificationPostfix(new_xyz, new_features_dc, new_features_rest, 
                            new_opacity, new_scaling, new_rotation);
    
    num_clone_operations_ += num_to_clone;
}

torch::Tensor CoarseToFineManager::getPruneMask(std::shared_ptr<GaussianModel>& pc)
{
    torch::Tensor opacity = pc->getOpacity();
    torch::Tensor scaling = pc->getScaling();
    
    // Prune low-opacity Gaussians
    torch::Tensor opacity_mask = opacity.squeeze() < params_.prune_opacity_threshold;
    
    // Prune oversized Gaussians
    torch::Tensor max_scale = std::get<0>(torch::max(scaling, 1));
    torch::Tensor scale_mask = max_scale > params_.prune_scale_threshold;
    
    return opacity_mask | scale_mask;
}

void CoarseToFineManager::applyPruneMask(std::shared_ptr<GaussianModel>& pc, 
                                        const torch::Tensor& prune_mask)
{
    torch::Tensor keep_mask = ~prune_mask;
    torch::Tensor indices = torch::nonzero(keep_mask).squeeze(1);
    
    // Update all Gaussian parameters
    pc->xyz_ = pc->xyz_.index_select(0, indices).detach().requires_grad_();
    pc->features_dc_ = pc->features_dc_.index_select(0, indices).detach().requires_grad_();
    pc->features_rest_ = pc->features_rest_.index_select(0, indices).detach().requires_grad_();
    pc->scaling_ = pc->scaling_.index_select(0, indices).detach().requires_grad_();
    pc->rotation_ = pc->rotation_.index_select(0, indices).detach().requires_grad_();
    pc->opacity_ = pc->opacity_.index_select(0, indices).detach().requires_grad_();
    
    // Update tensor vectors for optimizer
    pc->Tensor_vec_xyz_ = {pc->xyz_};
    pc->Tensor_vec_feature_dc_ = {pc->features_dc_};
    pc->Tensor_vec_feature_rest_ = {pc->features_rest_};
    pc->Tensor_vec_opacity_ = {pc->opacity_};
    pc->Tensor_vec_scaling_ = {pc->scaling_};
    pc->Tensor_vec_rotation_ = {pc->rotation_};
    pc->Tensor_vec_exposure_ = {pc->exposure_};
    
    // Reset optimizer state for pruned parameters
    if (pc->sparse_optimizer_) {
        // Note: In practice, you would need to update optimizer state
        // This is a simplified version
        pc->trainingSetup();  // Reinitialize optimizer
    }
}

// Global functions
void initializeCoarseToFineManager(const CoarseToFineParams& params)
{
    g_coarse_to_fine_manager = std::make_unique<CoarseToFineManager>(params);
}

CoarseToFineManager* getCoarseToFineManager()
{
    return g_coarse_to_fine_manager.get();
}

// ============================================================================
// AdaptiveVoxelManager Integration Methods
// ============================================================================

void CoarseToFineManager::setAdaptiveVoxelManager(std::shared_ptr<AdaptiveVoxelManager> voxel_manager)
{
    voxel_manager_ = voxel_manager;
    if (voxel_manager_) {
        std::cout << "\033[1;36m[CoarseToFine] Adaptive voxel manager enabled\033[0m" << std::endl;
    } else {
        std::cout << "\033[1;33m[CoarseToFine] Adaptive voxel manager disabled\033[0m" << std::endl;
    }
}

std::shared_ptr<AdaptiveVoxelManager> CoarseToFineManager::getAdaptiveVoxelManager() const
{
    return voxel_manager_;
}

void CoarseToFineManager::updateVoxelGrid(std::shared_ptr<GaussianModel>& pc, int iteration)
{
    if (!voxel_manager_ || !voxel_manager_->isEnabled()) {
        return;
    }
    
    // Only update voxel grid at specified intervals to avoid performance issues
    if (iteration % params_.voxel_update_interval != 0) {
        return;
    }
    
    try {
        // Get current Gaussian state
        auto positions = pc->getXYZ();
        
        // Check if gradients are available
        torch::Tensor gradients;
        if (pc->xyz_.grad().defined() && pc->xyz_.grad().numel() > 0) {
            gradients = pc->xyz_.grad();
        } else {
            // If no gradients available, create zero gradients
            gradients = torch::zeros_like(positions);
        }
        
        // 🔧 FIX: 计算真实法向量而不是使用零向量
        torch::Tensor normals;
        try {
            // 使用快速法向量估计
            if (gradients.sum().item<float>() != 0.0f) {
                // 如果有梯度信息，基于梯度估计法向量
                normals = FastNormalEstimator::estimateNormalsFromGradients(positions, gradients);
                std::cout << "[CoarseToFine] Using gradient-based normal estimation" << std::endl;
            } else {
                // 否则使用KNN方法估计法向量
                normals = FastNormalEstimator::estimateNormalsSimple(positions, 6);
                std::cout << "[CoarseToFine] Using KNN-based normal estimation" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "[CoarseToFine] Normal estimation failed: " << e.what() << ", using zero normals" << std::endl;
            normals = torch::zeros_like(positions);
        }
        
        // Create active indices
        std::vector<int> active_indices;
        int num_gaussians = positions.size(0);
        active_indices.reserve(num_gaussians);
        for (int i = 0; i < num_gaussians; ++i) {
            active_indices.push_back(i);
        }
        
        // Update voxel grid
        voxel_manager_->updateVoxelGrid(positions, gradients, normals, active_indices);
        
        // Log statistics periodically
        if (iteration % (params_.voxel_update_interval * 10) == 0) {
            auto stats = voxel_manager_->getStatistics();
            std::cout << "\033[1;36m[CoarseToFine] Voxel stats - Total: " << stats.total_voxels 
                      << ", Active: " << stats.active_voxels 
                      << ", Avg complexity: " << std::fixed << std::setprecision(3) << stats.average_complexity
                      << "\033[0m" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "\033[1;31m[CoarseToFine] Error updating voxel grid: " << e.what() << "\033[0m" << std::endl;
    }
}