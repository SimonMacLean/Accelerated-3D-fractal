#pragma once
#include "cuda_runtime.h"
struct fractal_creation_info
{
    float scale;
    float theta;
    float phi;
    float3 offset;
};
struct optimized_fractal_info
{
    float scale;
    float sin_theta;
    float cos_theta;
    float sin_phi;
    float cos_phi;
    float3 offset;
};
constexpr double pi = 3.14159265358979323846;
constexpr float minimum_distance = 0.001f;
constexpr float maximum_distance = 50;
constexpr int bytes = 3;
constexpr float ambient_occ_strength = 0.05f;
constexpr float shadow_strength = 8;
constexpr float side = 1;
constexpr float max_steps = 1024;