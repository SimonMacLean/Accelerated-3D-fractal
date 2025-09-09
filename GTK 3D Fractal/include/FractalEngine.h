#pragma once

#include "FractalParams.h"
#include <vector>
#include <cmath>

class FractalEngine {
public:
    FractalEngine();
    ~FractalEngine();
    
    // Render the fractal to a pixel buffer
    void render(std::vector<unsigned char>& pixel_buffer, int width, int height,
                const OptimizedFractalInfo& params, int iterations,
                const Vec3& camera_pos, const Vec3& light_pos);
    
    // Convert creation info to optimized info
    OptimizedFractalInfo optimizeParams(const FractalCreationInfo& info);
    
    // Get preset configuration
    FractalCreationInfo getPresetConfig(FractalPreset preset);
    
private:
    // Core fractal functions
    Vec3 abs_space(const Vec3& p);
    Vec3 max_space(const Vec3& a, const Vec3& b);
    Vec3 max_space(const Vec3& a, float b);
    Vec3 min_space(const Vec3& a, const Vec3& b);
    Vec3 min_space(const Vec3& a, float b);
    
    // Rotation functions
    Vec3 rotate_x(const Vec3& z, float s, float c);
    Vec3 rotate_y(const Vec3& z, float s, float c);
    Vec3 rotate_z(const Vec3& z, float s, float c);
    
    // Folding functions
    Vec3 fold_menger(Vec3 vec);
    Vec3 fold_box(const Vec3& z, float r);
    
    // Distance functions
    float distance_equation(const Vec3& p, float side);
    float scaled_de(Vec3 p, int iterations, float side, const OptimizedFractalInfo& params);
    
    // Transformation functions
    void transform(Vec3& p, const OptimizedFractalInfo& params);
    void warp_space(Vec3& p, int iterations, const OptimizedFractalInfo& params);
    float warp_scalar(float s, int iterations, float scale);
    
    // Lighting and shading
    Vec3 normal(const Vec3& p, int iterations, float side, 
                const OptimizedFractalInfo& params, float epsilon);
    float soft_shadow(const Vec3& p, const Vec3& d, float shadow_strength,
                     int iterations, float side, const OptimizedFractalInfo& params,
                     float min_dist, float max_dist, float min_angle);
    float orbit(Vec3 p, int iterations, float side, const OptimizedFractalInfo& params);
    
    // Color functions
    float trapezoid_wave(float loc);
    float red(float loc);
    float green(float loc);
    float blue(float loc);
    
    // Ray marching
    void march_ray(const Vec3& direction, unsigned char* pixel_rgb, 
                   const Vec3& camera, const Vec3& light, int iterations,
                   const OptimizedFractalInfo& params);
};