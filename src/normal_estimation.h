/*
 * 法向量估计工具
 * 
 * 基于KNN的法向量估计，为AdaptiveVoxelManager提供真实的几何信息
 */

#ifndef NORMAL_ESTIMATION_H
#define NORMAL_ESTIMATION_H

#include <torch/torch.h>
#include <vector>

/**
 * 法向量估计器
 */
class NormalEstimator {
public:
    explicit NormalEstimator(int k_neighbors = 10, float search_radius = 0.1f);
    ~NormalEstimator() = default;

    // 主要接口
    torch::Tensor estimateNormals(const torch::Tensor& positions);
    torch::Tensor estimateNormalsKNN(const torch::Tensor& positions, int k = 10);
    torch::Tensor estimateNormalsRadius(const torch::Tensor& positions, float radius = 0.1f);
    
    // 辅助方法
    torch::Tensor computeLocalCovariance(const torch::Tensor& local_points);
    torch::Tensor orientNormals(const torch::Tensor& normals, const torch::Tensor& positions);
    
    // 配置
    void setKNeighbors(int k) { k_neighbors_ = k; }
    void setSearchRadius(float radius) { search_radius_ = radius; }
    
private:
    // 核心算法
    std::vector<std::vector<int>> findKNearestNeighbors(const torch::Tensor& positions, int k);
    torch::Tensor computePCA(const torch::Tensor& points);
    torch::Tensor orientNormalConsistently(const torch::Tensor& normal, const torch::Tensor& viewpoint);
    
    // 参数
    int k_neighbors_;
    float search_radius_;
    bool use_viewpoint_orientation_;
};

/**
 * 快速法向量估计（简化版本）
 */
class FastNormalEstimator {
public:
    static torch::Tensor estimateNormalsSimple(const torch::Tensor& positions, int k = 6);
    static torch::Tensor estimateNormalsFromGradients(const torch::Tensor& positions, const torch::Tensor& gradients);
    static torch::Tensor estimateNormalsFromDepth(const torch::Tensor& positions, const torch::Tensor& depth_gradients);
};

#endif // NORMAL_ESTIMATION_H