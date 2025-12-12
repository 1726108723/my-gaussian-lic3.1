/*
 * Simple test for coarse-to-fine implementation
 */

#include "coarse_to_fine.h"
#include <iostream>

int main() {
    std::cout << "Testing Coarse-to-Fine Implementation..." << std::endl;
    
    // Test parameter creation
    CoarseToFineParams params;
    params.initial_sampling_ratio = 0.3f;
    params.densify_grad_threshold = 0.0002f;
    params.densify_size_threshold = 20.0f;
    params.prune_opacity_threshold = 0.005f;
    
    std::cout << "✓ Parameters created successfully" << std::endl;
    
    // Test manager initialization
    try {
        initializeCoarseToFineManager(params);
        std::cout << "✓ Manager initialized successfully" << std::endl;
        
        CoarseToFineManager* manager = getCoarseToFineManager();
        if (manager) {
            std::cout << "✓ Manager retrieved successfully" << std::endl;
            std::cout << "Current sampling ratio: " << manager->getCurrentSamplingRatio() << std::endl;
        } else {
            std::cout << "✗ Failed to retrieve manager" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cout << "✗ Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
}