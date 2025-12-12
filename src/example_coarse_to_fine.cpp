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

/**
 * @file example_coarse_to_fine.cpp
 * @brief Example demonstrating how to use the coarse-to-fine sampling strategy
 */

#include "gaussian.h"
#include "coarse_to_fine.h"
#include "mapping.h"

#include <iostream>
#include <memory>

/**
 * @brief Example function showing how to initialize and use coarse-to-fine strategy
 */
void demonstrateCoarseToFineUsage()
{
    std::cout << "\n=== Coarse-to-Fine Sampling Strategy Demo ===\n" << std::endl;
    
    // 1. Load parameters from config file (in practice, this would be done via YAML)
    // For this example, we'll create parameters manually
    YAML::Node config;
    config["height"] = 512;
    config["width"] = 640;
    config["fx"] = 431.71205;
    config["fy"] = 431.70855;
    config["cx"] = 320.3404;
    config["cy"] = 259.1696;
    config["select_every_k_frame"] = 5;
    config["sh_degree"] = 3;
    config["white_background"] = false;
    config["random_background"] = false;
    config["convert_SHs_python"] = false;
    config["compute_cov3D_python"] = false;
    config["lambda_erank"] = 0.0;
    config["scaling_scale"] = 1;
    config["position_lr"] = 0.00016;
    config["feature_lr"] = 0.0025;
    config["opacity_lr"] = 0.05;
    config["scaling_lr"] = 0.005;
    config["rotation_lr"] = 0.001;
    config["lambda_dssim"] = 0.2;
    config["apply_exposure"] = false;
    config["exposure_lr"] = 0.001;
    config["skybox_points_num"] = 100000;
    config["skybox_radius"] = 1000;
    
    // Coarse-to-fine parameters
    config["enable_coarse_to_fine"] = true;
    config["initial_sampling_ratio"] = 0.3f;
    config["min_initial_points"] = 1000;
    config["max_initial_points"] = 50000;
    config["densify_grad_threshold"] = 0.0002f;
    config["densify_size_threshold"] = 20.0f;
    config["densify_start_iter"] = 500;
    config["densify_stop_iter"] = 15000;
    config["densify_interval"] = 100;
    config["prune_opacity_threshold"] = 0.005f;
    config["prune_scale_threshold"] = 100.0f;
    config["prune_start_iter"] = 3000;
    config["prune_interval"] = 100;
    
    Params params(config);
    
    // 2. Initialize coarse-to-fine manager
    CoarseToFineParams ctf_params;
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
    
    initializeCoarseToFineManager(ctf_params);
    
    // 3. Create Gaussian model and dataset
    auto gaussian_model = std::make_shared<GaussianModel>(params);
    auto dataset = std::make_shared<Dataset>(params);
    
    // 4. In practice, you would add frames to the dataset here
    // For this example, we'll simulate some point cloud data
    std::cout << "Simulating point cloud data..." << std::endl;
    
    // Simulate 10000 random points
    for (int i = 0; i < 10000; ++i) {
        Eigen::Vector3d point(
            (rand() % 2000 - 1000) / 100.0,  // x: -10 to 10
            (rand() % 2000 - 1000) / 100.0,  // y: -10 to 10
            (rand() % 1000 + 100) / 100.0    // z: 1 to 11
        );
        Eigen::Vector3d color(
            rand() % 256 / 255.0,
            rand() % 256 / 255.0,
            rand() % 256 / 255.0
        );
        float depth = point.z();
        
        dataset->pointcloud_.push_back(point);
        dataset->pointcolor_.push_back(color);
        dataset->pointdepth_.push_back(depth);
    }
    
    std::cout << "Generated " << dataset->pointcloud_.size() << " points" << std::endl;
    
    // 5. Initialize with coarse-to-fine strategy
    if (params.enable_coarse_to_fine) {
        std::cout << "\nInitializing with coarse-to-fine strategy..." << std::endl;
        gaussian_model->initializeCoarseToFine(dataset);
    } else {
        std::cout << "\nInitializing with standard strategy..." << std::endl;
        gaussian_model->initialize(dataset);
    }
    
    // 6. Setup training
    gaussian_model->trainingSetup();
    
    // 7. Simulate training loop with coarse-to-fine refinement
    std::cout << "\nSimulating training loop..." << std::endl;
    
    for (int iteration = 0; iteration < 1000; iteration += 100) {
        // In practice, you would call optimize with real data
        // Here we just demonstrate the coarse-to-fine updates
        
        if (getCoarseToFineManager()) {
            // Note: updateRefinementLevel() has been removed in SplaTAM-style implementation
            // The manager now uses one-time initialization instead of progressive sampling
            
            // Simulate some gradients for demonstration
            if (iteration >= ctf_params.densify_start_iter && 
                iteration <= ctf_params.densify_stop_iter &&
                (iteration % ctf_params.densify_interval) == 0) {
                
                std::cout << "Iteration " << iteration << ": Applying densification..." << std::endl;
                getCoarseToFineManager()->adaptiveDensification(gaussian_model, iteration);
            }
            
            if (iteration >= ctf_params.prune_start_iter &&
                (iteration % ctf_params.prune_interval) == 0) {
                
                std::cout << "Iteration " << iteration << ": Applying pruning..." << std::endl;
                getCoarseToFineManager()->pruneGaussians(gaussian_model, iteration);
            }
        }
        
        std::cout << "Iteration " << iteration << ": Current Gaussian count: " 
                  << gaussian_model->getXYZ().size(0) << std::endl;
    }
    
    std::cout << "\n=== Demo completed successfully! ===\n" << std::endl;
    
    // Print final statistics
    std::cout << "Final Gaussian count: " << gaussian_model->getXYZ().size(0) << std::endl;
    // Note: getCurrentSamplingRatio() has been removed in SplaTAM-style implementation
    std::cout << "SplaTAM-style initialization: Uses all valid points at once" << std::endl;
}

/**
 * @brief Main function for the example
 */
int main(int argc, char** argv)
{
    try {
        demonstrateCoarseToFineUsage();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}