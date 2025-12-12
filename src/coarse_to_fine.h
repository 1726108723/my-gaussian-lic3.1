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

#pragma once

#include <torch/torch.h>
#include <memory>
#include <vector>

// Forward declarations
class GaussianModel;
class Dataset;
class AdaptiveVoxelManager;
class Camera;

/**
 * @brief Coarse-to-fine sampling strategy parameters
 */
struct CoarseToFineParams
{
    // Initial coarse sampling parameters
    float initial_sampling_ratio = 0.3f;        // Initial sampling ratio for coarse initialization
    int min_initial_points = 1000;              // Minimum number of initial points
    int max_initial_points = 50000;             // Maximum number of initial points
    
    // Adaptive densification parameters
    float densify_grad_threshold = 0.0002f;     // Gradient threshold for densification
    float densify_size_threshold = 20.0f;       // Size threshold for split vs clone
    int densify_start_iter = 500;               // Start densification after this many iterations
    int densify_stop_iter = 15000;              // Stop densification after this many iterations
    int densify_interval = 100;                 // Densification interval
    
    // Pruning parameters
    float prune_opacity_threshold = 0.005f;     // Opacity threshold for pruning
    float prune_scale_threshold = 100.0f;       // Scale threshold for pruning
    int prune_start_iter = 3000;                // Start pruning after this many iterations
    int prune_interval = 100;                   // Pruning interval
    
    // SplaTAM-style initialization parameters
    bool use_all_points_initialization = true;  // Use all valid points at once (SplaTAM style)
    float spatial_sampling_threshold = 0.01f;   // Minimum distance between points for spatial sampling
    
    // Adaptive voxel parameters
    int voxel_update_interval = 10;             // Voxel grid update interval
};

/**
 * @brief Coarse-to-fine sampling strategy manager
 */
class CoarseToFineManager
{
public:
    CoarseToFineManager(const CoarseToFineParams& params);
    
    /**
     * @brief Initialize Gaussian model with coarse sampling
     */
    void initializeCoarse(std::shared_ptr<GaussianModel>& pc, 
                         const std::shared_ptr<Dataset>& dataset);
    
    /**
     * @brief Perform adaptive densification based on gradients
     */
    void adaptiveDensification(std::shared_ptr<GaussianModel>& pc, 
                              int iteration);
    
    /**
     * @brief Prune low-opacity and oversized Gaussians
     */
    void pruneGaussians(std::shared_ptr<GaussianModel>& pc, 
                       int iteration);
    
    /**
     * @brief Add Gaussians based on silhouette/depth analysis (SplaTAM style)
     */
    void addSilhouetteGaussians(std::shared_ptr<GaussianModel>& pc, 
                               const std::shared_ptr<Camera>& viewpoint_camera,
                               const torch::Tensor& gt_image,
                               const torch::Tensor& gt_depth);
    
    /**
     * @brief Check if a point should be added at given position
     */
    bool shouldAddGaussianAt(const torch::Tensor& position) const;
    
    /**
     * @brief Check if densification should be performed
     */
    bool shouldDensify(int iteration) const;
    
    /**
     * @brief Check if pruning should be performed
     */
    bool shouldPrune(int iteration) const;
    
    /**
     * @brief Set adaptive voxel manager for geometry-aware processing
     */
    void setAdaptiveVoxelManager(std::shared_ptr<AdaptiveVoxelManager> voxel_manager);
    
    /**
     * @brief Get adaptive voxel manager
     */
    std::shared_ptr<AdaptiveVoxelManager> getAdaptiveVoxelManager() const;
    
    /**
     * @brief Update voxel grid with current Gaussian state
     */
    void updateVoxelGrid(std::shared_ptr<GaussianModel>& pc, int iteration);

private:
    CoarseToFineParams params_;
    int total_iterations_;
    
    // Adaptive voxel management
    std::shared_ptr<AdaptiveVoxelManager> voxel_manager_;
    
    // Statistics tracking
    int num_split_operations_;
    int num_clone_operations_;
    int num_pruned_gaussians_;
    
    /**
     * @brief Sample points based on current level
     */
    std::vector<int> samplePointIndices(int total_points, float sampling_ratio);
    
    /**
     * @brief Split large Gaussians with high gradients
     */
    void splitGaussians(std::shared_ptr<GaussianModel>& pc, 
                       const torch::Tensor& grad_mask,
                       const torch::Tensor& size_mask);
    
    /**
     * @brief Clone small Gaussians with high gradients
     */
    void cloneGaussians(std::shared_ptr<GaussianModel>& pc,
                       const torch::Tensor& grad_mask,
                       const torch::Tensor& size_mask);
    
    /**
     * @brief Remove Gaussians based on opacity and size thresholds
     */
    torch::Tensor getPruneMask(std::shared_ptr<GaussianModel>& pc);
    
    /**
     * @brief Apply pruning mask to remove Gaussians
     */
    void applyPruneMask(std::shared_ptr<GaussianModel>& pc, 
                       const torch::Tensor& prune_mask);
};

/**
 * @brief Global coarse-to-fine manager instance
 */
extern std::unique_ptr<CoarseToFineManager> g_coarse_to_fine_manager;

/**
 * @brief Initialize global coarse-to-fine manager
 */
void initializeCoarseToFineManager(const CoarseToFineParams& params);

/**
 * @brief Get global coarse-to-fine manager
 */
CoarseToFineManager* getCoarseToFineManager();