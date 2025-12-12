/*
 * 法向量估计工具实现
 */

#include "normal_estimation.h"
#include <iostream>
#include <algorithm>
#include <cmath>

// ============================================================================
// NormalEstimator 实现
// ============================================================================

NormalEstimator::NormalEstimator(int k_neighbors, float search_radius)
    : k_neighbors_(k_neighbors), search_radius_(search_radius), use_viewpoint_orientation_(true) {
}

torch::Tensor NormalEstimator::estimateNormals(const torch::Tensor& positions) {
    // 使用KNN方法作为默认
    return estimateNormalsKNN(positions, k_neighbors_);
}

torch::Tensor NormalEstimator::estimateNormalsKNN(const torch::Tensor& positions, int k) {
    if (positions.size(0) < k) {
        std::cout << "[NormalEstimator] Warning: Not enough points for KNN, using simplified estimation" << std::endl;
        return FastNormalEstimator::estimateNormalsSimple(positions, std::min(k, (int)positions.size(0) - 1));
    }
    
    int num_points = positions.size(0);
    torch::Tensor normals = torch::zeros({num_points, 3}, positions.options());
    
    // 找到每个点的K近邻
    auto neighbors = findKNearestNeighbors(positions, k);
    
    for (int i = 0; i < num_points; ++i) {
        if (neighbors[i].size() < 3) {
            // 如果邻居太少，使用默认法向量
            normals[i] = torch::tensor({0.0f, 0.0f, 1.0f}, positions.options());
            continue;
        }
        
        // 收集邻居点
        torch::Tensor neighbor_positions = torch::zeros({(int)neighbors[i].size(), 3}, positions.options());
        for (size_t j = 0; j < neighbors[i].size(); ++j) {
            neighbor_positions[j] = positions[neighbors[i][j]];
        }
        
        // 计算局部协方差矩阵并进行PCA
        torch::Tensor normal = computePCA(neighbor_positions);
        normals[i] = normal;
    }
    
    // 法向量方向一致性调整
    normals = orientNormals(normals, positions);
    
    return normals;
}

std::vector<std::vector<int>> NormalEstimator::findKNearestNeighbors(const torch::Tensor& positions, int k) {
    int num_points = positions.size(0);
    std::vector<std::vector<int>> neighbors(num_points);
    
    for (int i = 0; i < num_points; ++i) {
        std::vector<std::pair<float, int>> distances;
        
        for (int j = 0; j < num_points; ++j) {
            if (i == j) continue;
            
            torch::Tensor diff = positions[i] - positions[j];
            float dist = torch::norm(diff).item<float>();
            distances.push_back({dist, j});
        }
        
        // 排序并取前K个
        std::sort(distances.begin(), distances.end());
        
        int actual_k = std::min(k, (int)distances.size());
        neighbors[i].reserve(actual_k);
        for (int j = 0; j < actual_k; ++j) {
            neighbors[i].push_back(distances[j].second);
        }
    }
    
    return neighbors;
}

torch::Tensor NormalEstimator::computePCA(const torch::Tensor& points) {
    if (points.size(0) < 3) {
        return torch::tensor({0.0f, 0.0f, 1.0f}, points.options());
    }
    
    // 计算质心
    torch::Tensor centroid = torch::mean(points, 0);
    
    // 中心化点云
    torch::Tensor centered = points - centroid.unsqueeze(0);
    
    // 计算协方差矩阵
    torch::Tensor covariance = torch::mm(centered.transpose(0, 1), centered) / (points.size(0) - 1);
    
    // 计算特征值和特征向量
    auto [eigenvalues, eigenvectors] = torch::linalg_eigh(covariance);
    
    // 最小特征值对应的特征向量就是法向量
    torch::Tensor normal = eigenvectors.select(1, 0);  // 第一列（最小特征值）
    
    // 归一化
    normal = normal / torch::norm(normal);
    
    return normal;
}

torch::Tensor NormalEstimator::orientNormals(const torch::Tensor& normals, const torch::Tensor& positions) {
    // 简单的法向量方向调整：确保指向外部
    torch::Tensor oriented_normals = normals.clone();
    
    if (use_viewpoint_orientation_) {
        // 假设视点在原点，法向量应该指向视点
        torch::Tensor centroid = torch::mean(positions, 0);
        
        for (int i = 0; i < normals.size(0); ++i) {
            torch::Tensor to_viewpoint = -positions[i];  // 指向原点
            float dot_product = torch::dot(normals[i], to_viewpoint).item<float>();
            
            if (dot_product < 0) {
                oriented_normals[i] = -normals[i];
            }
        }
    }
    
    return oriented_normals;
}

// ============================================================================
// FastNormalEstimator 实现
// ============================================================================

torch::Tensor FastNormalEstimator::estimateNormalsSimple(const torch::Tensor& positions, int k) {
    int num_points = positions.size(0);
    if (num_points < 3 || k < 2) {
        // 返回默认法向量
        return torch::zeros({num_points, 3}, positions.options()) + torch::tensor({0.0f, 0.0f, 1.0f}, positions.options());
    }
    
    torch::Tensor normals = torch::zeros({num_points, 3}, positions.options());
    
    for (int i = 0; i < num_points; ++i) {
        // 计算到所有其他点的距离
        torch::Tensor distances = torch::norm(positions - positions[i].unsqueeze(0), 2, 1);
        
        // 找到最近的k个点（排除自己）
        auto [sorted_distances, indices] = torch::sort(distances);
        
        // 取前k+1个（包括自己），然后排除自己
        int actual_k = std::min(k + 1, num_points);
        torch::Tensor neighbor_indices = indices.slice(0, 1, actual_k);  // 排除第0个（自己）
        
        if (neighbor_indices.size(0) < 2) {
            normals[i] = torch::tensor({0.0f, 0.0f, 1.0f}, positions.options());
            continue;
        }
        
        // 收集邻居点
        torch::Tensor neighbors = torch::index_select(positions, 0, neighbor_indices);
        
        // 简化的法向量计算：使用前两个邻居构成的向量的叉积
        if (neighbors.size(0) >= 2) {
            torch::Tensor v1 = neighbors[0] - positions[i];
            torch::Tensor v2 = neighbors[1] - positions[i];
            
            torch::Tensor normal = torch::cross(v1, v2);
            float norm = torch::norm(normal).item<float>();
            
            if (norm > 1e-6) {
                normals[i] = normal / norm;
            } else {
                normals[i] = torch::tensor({0.0f, 0.0f, 1.0f}, positions.options());
            }
        } else {
            normals[i] = torch::tensor({0.0f, 0.0f, 1.0f}, positions.options());
        }
    }
    
    return normals;
}

torch::Tensor FastNormalEstimator::estimateNormalsFromGradients(const torch::Tensor& positions, const torch::Tensor& gradients) {
    // 基于梯度信息估计法向量
    // 梯度方向通常垂直于表面，可以作为法向量的近似
    
    torch::Tensor normals = gradients.clone();
    
    // 归一化
    torch::Tensor norms = torch::norm(normals, 2, 1, true);
    norms = torch::clamp(norms, 1e-6);  // 避免除零
    normals = normals / norms;
    
    return normals;
}

torch::Tensor FastNormalEstimator::estimateNormalsFromDepth(const torch::Tensor& positions, const torch::Tensor& depth_gradients) {
    // 基于深度梯度估计法向量
    // 这是一个简化实现，假设深度梯度提供了表面方向信息
    
    if (depth_gradients.size(1) < 2) {
        // 如果没有足够的梯度信息，返回默认法向量
        return torch::zeros({positions.size(0), 3}, positions.options()) + 
               torch::tensor({0.0f, 0.0f, 1.0f}, positions.options());
    }
    
    torch::Tensor normals = torch::zeros({positions.size(0), 3}, positions.options());
    
    // 使用深度梯度的x和y分量构造法向量
    normals.select(1, 0) = -depth_gradients.select(1, 0);  // -dx
    normals.select(1, 1) = -depth_gradients.select(1, 1);  // -dy
    normals.select(1, 2) = torch::ones({positions.size(0)}, positions.options());  // 1
    
    // 归一化
    torch::Tensor norms = torch::norm(normals, 2, 1, true);
    norms = torch::clamp(norms, 1e-6);
    normals = normals / norms;
    
    return normals;
}