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

#include "yaml_utils.h"

#include <chrono>
#include <deque>
#include <queue>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

#include <geometry_msgs/PoseStamped.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <eigen_conversions/eigen_msg.h>
#include <Eigen/Eigen>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

class Params
{
public:
    Params(const YAML::Node &node)
    {
        height = node["height"].as<int>();
        width = node["width"].as<int>();
        fx = node["fx"].as<double>();
        fy = node["fy"].as<double>();
        cx = node["cx"].as<double>();
        cy = node["cy"].as<double>();

        select_every_k_frame = node["select_every_k_frame"].as<int>();

        sh_degree = node["sh_degree"].as<int>();
        white_background = node["white_background"].as<bool>();
        random_background = node["random_background"].as<bool>();
        convert_SHs_python = node["convert_SHs_python"].as<bool>();
        compute_cov3D_python = node["compute_cov3D_python"].as<bool>();
        lambda_erank = node["lambda_erank"].as<double>();
        scaling_scale = node["scaling_scale"].as<int>();

        position_lr = node["position_lr"].as<double>();
        feature_lr = node["feature_lr"].as<double>();
        opacity_lr = node["opacity_lr"].as<double>();
        scaling_lr = node["scaling_lr"].as<double>();
        rotation_lr = node["rotation_lr"].as<double>();
        lambda_dssim = node["lambda_dssim"].as<double>();

        apply_exposure = node["apply_exposure"].as<bool>();
        exposure_lr = node["exposure_lr"].as<double>();
        skybox_points_num = node["skybox_points_num"].as<int>();
        skybox_radius = node["skybox_radius"].as<int>();
        
        // Coarse-to-fine parameters (with default values)
        enable_coarse_to_fine = node["enable_coarse_to_fine"] ? node["enable_coarse_to_fine"].as<bool>() : false;
        initial_sampling_ratio = node["initial_sampling_ratio"] ? node["initial_sampling_ratio"].as<float>() : 0.3f;
        min_initial_points = node["min_initial_points"] ? node["min_initial_points"].as<int>() : 1000;
        max_initial_points = node["max_initial_points"] ? node["max_initial_points"].as<int>() : 50000;
        densify_grad_threshold = node["densify_grad_threshold"] ? node["densify_grad_threshold"].as<float>() : 0.0002f;
        densify_size_threshold = node["densify_size_threshold"] ? node["densify_size_threshold"].as<float>() : 20.0f;
        densify_start_iter = node["densify_start_iter"] ? node["densify_start_iter"].as<int>() : 500;
        densify_stop_iter = node["densify_stop_iter"] ? node["densify_stop_iter"].as<int>() : 15000;
        densify_interval = node["densify_interval"] ? node["densify_interval"].as<int>() : 100;
        prune_opacity_threshold = node["prune_opacity_threshold"] ? node["prune_opacity_threshold"].as<float>() : 0.005f;
        prune_scale_threshold = node["prune_scale_threshold"] ? node["prune_scale_threshold"].as<float>() : 100.0f;
        prune_start_iter = node["prune_start_iter"] ? node["prune_start_iter"].as<int>() : 3000;
        prune_interval = node["prune_interval"] ? node["prune_interval"].as<int>() : 100;
        
        // Advanced coarse-to-fine parameters (optional)
        refinement_levels = node["refinement_levels"] ? node["refinement_levels"].as<int>() : 2;
        adaptive_sampling = node["adaptive_sampling"] ? node["adaptive_sampling"].as<bool>() : false;
        gradient_accumulation_window = node["gradient_accumulation_window"] ? node["gradient_accumulation_window"].as<int>() : 30;
        
        // Adaptive Voxel Manager parameters (optional)
        enable_adaptive_voxel = node["enable_adaptive_voxel"] ? node["enable_adaptive_voxel"].as<bool>() : false;
        base_voxel_size = node["base_voxel_size"] ? node["base_voxel_size"].as<float>() : 0.1f;
        max_voxel_levels = node["max_voxel_levels"] ? node["max_voxel_levels"].as<int>() : 4;
        complexity_threshold_refine = node["complexity_threshold_refine"] ? node["complexity_threshold_refine"].as<float>() : 0.7f;
        complexity_threshold_coarsen = node["complexity_threshold_coarsen"] ? node["complexity_threshold_coarsen"].as<float>() : 0.3f;
        voxel_update_interval = node["voxel_update_interval"] ? node["voxel_update_interval"].as<int>() : 10;
        gradient_weight = node["gradient_weight"] ? node["gradient_weight"].as<float>() : 0.4f;
        curvature_weight = node["curvature_weight"] ? node["curvature_weight"].as<float>() : 0.4f;
        density_weight = node["density_weight"] ? node["density_weight"].as<float>() : 0.2f;
    }

    /// dataset
    int height;
    int width;
    double fx;
    double fy;
    double cx;
    double cy;

    int select_every_k_frame;

    /// gaussian
    int sh_degree;
    bool white_background;
    bool random_background;
    bool convert_SHs_python;
    bool compute_cov3D_python;
    float lambda_erank;
    double scaling_scale;

    double position_lr;
    double feature_lr;
    double opacity_lr;
    double scaling_lr;
    double rotation_lr;
    double lambda_dssim;

    bool apply_exposure;
    double exposure_lr;
    int skybox_points_num;
    int skybox_radius;
    
    // Coarse-to-fine parameters
    bool enable_coarse_to_fine;
    float initial_sampling_ratio;
    int min_initial_points;
    int max_initial_points;
    float densify_grad_threshold;
    float densify_size_threshold;
    int densify_start_iter;
    int densify_stop_iter;
    int densify_interval;
    float prune_opacity_threshold;
    float prune_scale_threshold;
    int prune_start_iter;
    int prune_interval;
    
    // Advanced coarse-to-fine parameters
    int refinement_levels;
    bool adaptive_sampling;
    int gradient_accumulation_window;
    
    // Adaptive Voxel Manager parameters
    bool enable_adaptive_voxel;
    float base_voxel_size;
    int max_voxel_levels;
    float complexity_threshold_refine;
    float complexity_threshold_coarsen;
    int voxel_update_interval;
    float gradient_weight;
    float curvature_weight;
    float density_weight;
};

struct Frame 
{
    sensor_msgs::PointCloud2ConstPtr point_msg;
    geometry_msgs::PoseStampedConstPtr pose_msg;
    sensor_msgs::ImageConstPtr image_msg;
};