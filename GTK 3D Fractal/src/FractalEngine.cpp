#include "FractalEngine.h"
#include <algorithm>
#include <cmath>

FractalEngine::FractalEngine() {
}

FractalEngine::~FractalEngine() {
}

Vec3 FractalEngine::abs_space(const Vec3& p) {
    return Vec3(std::abs(p.x), std::abs(p.y), std::abs(p.z));
}

Vec3 FractalEngine::max_space(const Vec3& a, const Vec3& b) {
    return Vec3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}

Vec3 FractalEngine::max_space(const Vec3& a, float b) {
    return Vec3(std::max(a.x, b), std::max(a.y, b), std::max(a.z, b));
}

Vec3 FractalEngine::min_space(const Vec3& a, const Vec3& b) {
    return Vec3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}

Vec3 FractalEngine::min_space(const Vec3& a, float b) {
    return Vec3(std::min(a.x, b), std::min(a.y, b), std::min(a.z, b));
}

Vec3 FractalEngine::rotate_x(const Vec3& z, float s, float c) {
    return Vec3(z.x, c * z.y + s * z.z, c * z.z - s * z.y);
}

Vec3 FractalEngine::rotate_y(const Vec3& z, float s, float c) {
    return Vec3(c * z.x - s * z.z, z.y, c * z.z + s * z.x);
}

Vec3 FractalEngine::rotate_z(const Vec3& z, float s, float c) {
    return Vec3(c * z.x + s * z.y, c * z.y - s * z.x, z.z);
}

Vec3 FractalEngine::fold_menger(Vec3 vec) {
    float a = std::min(vec.x - vec.y, 0.0f);
    vec.x -= a;
    vec.y += a;
    a = std::min(vec.x - vec.z, 0.0f);
    vec.x -= a;
    vec.z += a;
    a = std::min(vec.y - vec.z, 0.0f);
    vec.y -= a;
    vec.z += a;
    return vec;
}

Vec3 FractalEngine::fold_box(const Vec3& z, float r) {
    return max_space(min_space(z, r), -r) * 2.0f - z;
}

float FractalEngine::distance_equation(const Vec3& p, float side) {
    return std::max(std::abs(p.x), std::max(std::abs(p.y), std::abs(p.z))) - side / 2.0f;
}

void FractalEngine::transform(Vec3& p, const OptimizedFractalInfo& params) {
    p = p / params.scale;
    p = abs_space(p);
    p = rotate_z(p, params.sin_theta, params.cos_theta);
    p = fold_menger(p);
    p = rotate_x(p, params.sin_phi, params.cos_phi);
    p = p - params.offset;
}

void FractalEngine::warp_space(Vec3& p, int iterations, const OptimizedFractalInfo& params) {
    for (int i = 0; i < iterations; ++i) {
        transform(p, params);
    }
}

float FractalEngine::warp_scalar(float s, int iterations, float scale) {
    for (int i = 0; i < iterations; ++i) {
        s *= scale;
    }
    return s;
}

float FractalEngine::scaled_de(Vec3 p, int iterations, float side, const OptimizedFractalInfo& params) {
    warp_space(p, iterations, params);
    return warp_scalar(distance_equation(p, side), iterations, params.scale);
}

Vec3 FractalEngine::normal(const Vec3& p, int iterations, float side, 
                          const OptimizedFractalInfo& params, float epsilon) {
    float scaledde = scaled_de(p, iterations, side, params);
    Vec3 scaled = Vec3(
        scaled_de(Vec3(p.x + epsilon, p.y, p.z), iterations, side, params) - scaledde,
        scaled_de(Vec3(p.x, p.y + epsilon, p.z), iterations, side, params) - scaledde,
        scaled_de(Vec3(p.x, p.y, p.z + epsilon), iterations, side, params) - scaledde
    );
    return scaled.normalized();
}

float FractalEngine::soft_shadow(const Vec3& p, const Vec3& d, float shadow_strength,
                                 int iterations, float side, const OptimizedFractalInfo& params,
                                 float min_dist, float max_dist, float min_angle) {
    float darkness = 1.0f;
    float prev_dist = 2147483647.0f;
    float total_dist = min_dist;
    
    while (total_dist < max_dist) {
        float dist = scaled_de(p + d * total_dist, iterations, side, params);
        float old_new_int_dist = dist * dist / (2.0f * prev_dist);
        float leg_length = std::sqrt(dist * dist - old_new_int_dist * old_new_int_dist);
        float angle = shadow_strength * leg_length / std::max(0.0f, total_dist - old_new_int_dist);
        darkness = std::min(darkness, angle);
        prev_dist = dist;
        total_dist += dist;
        if (dist <= 0 || darkness < min_angle) {
            return 0.0f;
        }
    }
    return darkness;
}

float FractalEngine::orbit(Vec3 p, int iterations, float side, const OptimizedFractalInfo& params) {
    float direction = scaled_de(p, 1, side, params);
    for (int i = 0; i < iterations; ++i) {
        transform(p, params);
        float f = warp_scalar(distance_equation(p, side), i, params.scale);
        if (f * direction >= 0) {
            return f / params.scale * 6.0f;
        }
    }
    return warp_scalar(distance_equation(p, side), iterations - 1, params.scale) * 6.0f;
}

float FractalEngine::trapezoid_wave(float loc) {
    return std::min(std::max(std::abs(loc - 3.0f), 0.0f) - 1.0f, 1.0f);
}

float FractalEngine::red(float loc) {
    return trapezoid_wave(std::fmod(loc, 6.0f));
}

float FractalEngine::green(float loc) {
    return trapezoid_wave(std::fmod(loc + 4.0f, 6.0f));
}

float FractalEngine::blue(float loc) {
    return trapezoid_wave(std::fmod(loc + 2.0f, 6.0f));
}

void FractalEngine::march_ray(const Vec3& direction, unsigned char* pixel_rgb, 
                              const Vec3& camera, const Vec3& light, int iterations,
                              const OptimizedFractalInfo& params) {
    Vec3 current_pos = camera;
    float total_distance = 0.0f;
    
    for (int step = 0; step < MAX_STEPS; ++step) {
        float dist = scaled_de(current_pos, iterations, SIDE, params);
        
        if (dist < MINIMUM_DISTANCE) {
            // Hit the fractal surface
            Vec3 normal_vec = normal(current_pos, iterations, SIDE, params, MINIMUM_DISTANCE);
            Vec3 light_dir = (light - current_pos).normalized();
            
            // Basic lighting calculation
            float diffuse = std::max(0.0f, normal_vec.dot(light_dir));
            
            // Soft shadows
            float shadow = soft_shadow(current_pos, light_dir, SHADOW_STRENGTH, 
                                     iterations, SIDE, params, MINIMUM_DISTANCE, 
                                     MAXIMUM_DISTANCE, 0.1f);
            
            // Orbit coloring
            float orbit_value = orbit(current_pos, iterations, SIDE, params);
            
            // Combine lighting and color
            float r = red(orbit_value) * diffuse * shadow + AMBIENT_OCC_STRENGTH;
            float g = green(orbit_value) * diffuse * shadow + AMBIENT_OCC_STRENGTH;
            float b = blue(orbit_value) * diffuse * shadow + AMBIENT_OCC_STRENGTH;
            
            pixel_rgb[0] = static_cast<unsigned char>(std::clamp(r * 255.0f, 0.0f, 255.0f));
            pixel_rgb[1] = static_cast<unsigned char>(std::clamp(g * 255.0f, 0.0f, 255.0f));
            pixel_rgb[2] = static_cast<unsigned char>(std::clamp(b * 255.0f, 0.0f, 255.0f));
            return;
        }
        
        if (total_distance > MAXIMUM_DISTANCE) {
            // Ray escaped - set background color
            pixel_rgb[0] = 0;
            pixel_rgb[1] = 0;
            pixel_rgb[2] = 0;
            return;
        }
        
        current_pos += direction * dist;
        total_distance += dist;
    }
    
    // Max steps reached - set background color
    pixel_rgb[0] = 0;
    pixel_rgb[1] = 0;
    pixel_rgb[2] = 0;
}

void FractalEngine::render(std::vector<unsigned char>& pixel_buffer, int width, int height,
                          const OptimizedFractalInfo& params, int iterations,
                          const Vec3& camera_pos, const Vec3& light_pos) {
    pixel_buffer.resize(width * height * 3);
    
    // Camera setup
    Vec3 camera_forward = Vec3(0, 0, 1);
    Vec3 camera_right = Vec3(1, 0, 0);
    Vec3 camera_up = Vec3(0, 1, 0);
    float focal_length = 1.0f;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Calculate ray direction
            float screen_x = (x - width / 2.0f) / height;
            float screen_y = (y - height / 2.0f) / height;
            
            Vec3 ray_dir = (camera_forward * focal_length + 
                           camera_right * screen_x + 
                           camera_up * screen_y).normalized();
            
            int pixel_index = (y * width + x) * 3;
            march_ray(ray_dir, &pixel_buffer[pixel_index], camera_pos, light_pos, iterations, params);
        }
    }
}

OptimizedFractalInfo FractalEngine::optimizeParams(const FractalCreationInfo& info) {
    OptimizedFractalInfo optimized;
    optimized.scale = info.scale;
    optimized.sin_theta = std::sin(info.theta);
    optimized.cos_theta = std::cos(info.theta);
    optimized.sin_phi = std::sin(info.phi);
    optimized.cos_phi = std::cos(info.phi);
    optimized.offset = info.offset;
    return optimized;
}

FractalCreationInfo FractalEngine::getPresetConfig(FractalPreset preset) {
    FractalCreationInfo config;
    
    // Default values
    config.scale = 1.0f;
    config.theta = 0.0f;
    config.phi = 0.0f;
    config.offset = Vec3(0.0f, 0.0f, 0.0f);
    
    // Apply preset-specific configurations
    switch (preset) {
        case FractalPreset::JUMP_THE_CRATER:
            config.scale = 1.2f;
            config.theta = 0.5f;
            config.phi = 0.3f;
            config.offset = Vec3(0.1f, 0.1f, 0.0f);
            break;
        case FractalPreset::TOO_MANY_TREES:
            config.scale = 0.8f;
            config.theta = 1.0f;
            config.phi = 0.8f;
            config.offset = Vec3(0.2f, 0.0f, 0.1f);
            break;
        case FractalPreset::HOLE_IN_ONE:
            config.scale = 1.5f;
            config.theta = 0.2f;
            config.phi = 1.2f;
            config.offset = Vec3(0.0f, 0.3f, 0.2f);
            break;
        // Add more presets as needed
        default:
            break;
    }
    
    return config;
}