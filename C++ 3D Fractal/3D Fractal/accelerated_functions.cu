#include "device_launch_parameters.h"
#include "fractal_params.h"
#include <algorithm>
#include <cuda_runtime.h>

using namespace std;
__device__ inline float operator !(const float3 v)
{
    return rsqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}
__device__ inline float operator &(const float3 v1, const float3 v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}
__device__ inline float3 operator ^(const float3 v1, const float3 v2)
{
    return make_float3(
        v1.y * v2.z - v1.z * v2.y, 
        v1.z * v2.x - v1.x * v2.z, 
        v1.x * v2.y - v1.y * v2.x );
}
__device__ inline float3 operator /(float3 v1, const float3 v2)
{
    v1.x /= v2.x;
    v1.y /= v2.y;
    v1.z /= v2.z;
    return v1;
}
__device__ inline float3 operator /(float3 v, const float s)
{
    v.x /= s;
    v.y /= s;
    v.z /= s;
    return v;
}
__device__ inline float3 operator *(float3 v1, const float3 v2)
{
    v1.x *= v2.x;
    v1.y *= v2.y;
    v1.z *= v2.z;
    return v1;
}
__device__ inline float3 operator *(float3 v, const float s)
{
    v.x *= s;
    v.y *= s;
    v.z *= s;
    return v;
}
__device__ inline float3 operator +(float3 v1, const float3 v2)
{
    v1.x += v2.x;
    v1.y += v2.y;
    v1.z += v2.z;
    return v1;
}
__device__ inline float3 operator +(float3 v, const float s)
{
    v.x += s;
    v.y += s;
    v.z += s;
    return v;
}
__device__ inline float3 operator -(float3 v1, const float3 v2)
{
    v1.x -= v2.x;
    v1.y -= v2.y;
    v1.z -= v2.z;
    return v1;
}
__device__ inline float3 operator -(float3 v, const float s)
{
    v.x -= s;
    v.y -= s;
    v.z -= s;
    return v;
}
__device__ inline float3 operator %(float3 v1, const float3 v2)
{
    v1.x = fmodf(v1.x, v2.x);
    v1.y = fmodf(v1.y, v2.y);
    v1.z = fmodf(v1.z, v2.z);
    return v1;
}
__device__ inline int index(const int x, const int y, const int width)
{
    return x + y * width;
}
__device__ inline float3 rotate_vec(const float3 vec, const float3 axis, const float cos, const float sin)
{
    const float d = (1 - cos) * (axis & vec);
    return make_float3(
        d * axis.x + vec.x * cos + sin * (axis.y * vec.z - axis.z * vec.y), 
        d * axis.y + vec.y * cos + sin * (axis.z * vec.x - axis.x * vec.z), 
        d * axis.z + vec.z * cos + sin * (axis.x * vec.y - axis.y * vec.x));
}
__device__ inline float warp_scalar(float s, const int iterations, const float scale)
{
    for (int i = 0; i < iterations; ++i)
        s *= scale;
    return s;
}
__device__ inline float cube_distance_equation(const float3 p, const float3 cube_center, const float cube_side_length)
{
    return max(abs(p.x - cube_center.x), max(abs(p.y - cube_center.y), abs(p.z - cube_center.z))) - cube_side_length / 2;
}
__device__ inline float distance_equation(const float3 p, const float side)
{
    return fmaxf(fabsf(p.x), fmaxf(fabsf(p.y), fabsf(p.z))) - side / 2;
}
__device__ inline float3 mod_space(const float3 p, const float3 mod)
{
    return ((p + mod / 2) % mod + mod) % mod - mod / 2;
}
__device__ inline float3 fold_space(const float3 p, const float3 n)
{
    if ((p & n) >= 0)
        return p;
    return p - n * 2 * (p & n) / (n & n);
}
__device__ inline float3 fold_menger(float3 vec)
{
    float a = fminf(vec.x - vec.y, 0.f);
    vec.x -= a;
    vec.y += a;
    a = fminf(vec.x - vec.z, 0.f);
    vec.x -= a;
    vec.z += a;
    a = fminf(vec.y - vec.z, 0.f);
    vec.y -= a;
    vec.z += a;
    return vec;
}
__device__ inline float3 max_space(float3 a, const float3 b)
{
    a.x = fmaxf(a.x, b.x);
    a.y = fmaxf(a.y, b.y);
	a.z = fmaxf(a.z, b.z);
    return a;
}
__device__ inline float3 max_space(float3 a, const float b)
{
    a.x = fmaxf(a.x, b);
    a.y = fmaxf(a.y, b);
    a.z = fmaxf(a.z, b);
    return a;
}
__device__ inline float3 min_space(float3 a, const float3 b)
{
    a.x = fminf(a.x, b.x);
    a.y = fminf(a.y, b.y);
    a.z = fminf(a.z, b.z);
    return a;
}
__device__ inline float3 min_space(float3 a, const float b)
{
    a.x = fminf(a.x, b);
    a.y = fminf(a.y, b);
    a.z = fminf(a.z, b);
    return a;
}
__device__ inline float3 fold_box(const float3 z, const float r)
{
    return max_space(min_space(z, r), -r) * 2 - z;
}
__device__ inline float3 abs_space(float3 p)
{
    p.x = fabsf(p.x);
    p.y = fabsf(p.y);
    p.z = fabsf(p.z);
    return p;
}
__device__ inline float3 abs_space_x(float3 p)
{
    p.x = fabsf(p.x);
    return p;
}
__device__ inline float3 abs_space_y(float3 p)
{
    p.y = fabsf(p.y);
    return p;
}
__device__ inline float3 abs_space_z(float3 p)
{
    p.z = fabsf(p.z);
    return p;
}
__device__ inline float3 rotate_x(const float3 z, const float s, const float c)
{
    return make_float3(z.x, c * z.y + s * z.z,c * z.z - s * z.y);
}
__device__ inline float3 rotate_y(const float3 z, const float s, const float c)
{
    return make_float3(c * z.x - s * z.z, z.y,c * z.z + s * z.x);
}
__device__ inline float3 rotate_z(const float3 z, const float s, const float c)
{
    return make_float3(c * z.x + s * z.y,c * z.y - s * z.x, z.z);
}
__device__ inline void transform(float3& p, const optimized_fractal_info params)
{
    p = p / params.scale;
    p = abs_space(p);
    p = rotate_z(p,params.sin_theta, params.cos_theta);
    p = fold_menger(p);
    p = rotate_x(p, params.sin_phi, params.cos_phi);
    p = p - params.offset;
}
__device__ inline void warp_space(float3& p, const int iterations, const optimized_fractal_info params)
{
    for (int i = 0; i < iterations; ++i)
        transform(p, params);
}
__device__ inline float scaled_de(float3 p, const int iterations, const float side, const optimized_fractal_info params)
{
    warp_space(p, iterations, params);
    return warp_scalar(distance_equation(p, side), iterations, params.scale);
}
__device__ inline float3 normal(const float3 p, const int iterations, const float side, const optimized_fractal_info params, const float epsilon)
{
    float scaledde = scaled_de(make_float3(p.x, p.y, p.z), iterations, side, params);
    const float3 scaled = {
        scaled_de(make_float3(p.x + epsilon, p.y, p.z), iterations, side, params) -
        scaledde,
        scaled_de(make_float3(p.x, p.y + epsilon, p.z), iterations, side, params) -
        scaledde,
        scaled_de(make_float3(p.x, p.y, p.z + epsilon), iterations, side, params) -
        scaledde
    };
    return scaled * !scaled;
}
__device__ inline float soft_shadow(const float3 p, const float3 d, const float shadow_strength, const int iterations, const float side, const optimized_fractal_info params, const float minDist, const float maxDist, const float minAngle)
{
    float darkness = 1;
    float prev_dist = 2147483647;
    float total_dist = minDist;
    while (total_dist < maxDist)
    {
        const float dist = scaled_de(p + d * total_dist, iterations, side, params);
        const float old_new_int_dist = dist * dist / (2 * prev_dist);
        const float leg_length = sqrtf(dist * dist - old_new_int_dist * old_new_int_dist);
        float angle = shadow_strength * leg_length / fmaxf(0.f, total_dist - old_new_int_dist);
        darkness = fminf(darkness, angle);
        prev_dist = dist;
        total_dist += dist;
        if (dist <= 0 || darkness < minAngle)
            return 0;
    }
    return darkness;
}
__device__ inline float orbit(float3 p, const int iterations, const float side, const optimized_fractal_info params)
{
    const float direction = scaled_de(p, 1, side, params);
    for (int i = 0; i < iterations; ++i)
    {
        transform(p, params);
        const float f = warp_scalar(distance_equation(p, side), i, params.scale);
        if (f * direction >= 0)
            return f / params.scale * 6;
    }
    return warp_scalar(distance_equation(p, side), iterations - 1, params.scale) * 6;
}
__device__ inline float trapezoid_wave(const float loc)
{
    return fminf(fmaxf(fabsf(loc - 3), 0) - 1, 1);
}
__device__ inline float red(const float loc)
{
    return trapezoid_wave(fmodf(loc, 6));
}
__device__ inline float green(const float loc)
{
    return trapezoid_wave(fmodf(loc + 4, 6));
}
__device__ inline float blue(const float loc)
{
    return trapezoid_wave(fmodf(loc + 2, 6));
}
__global__ void get_direction_length_inv(float* ray_length_invs, const float focal_length, const int width, const int height)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height)
        return;
    ray_length_invs[index(i, height - 1 - j, width)] = rsqrtf(
        focal_length * focal_length + ((j - height / 2.f) * (j - height / 2.f) + (i - width / 2.f) * (i - width / 2.f)) / height / height);
}
__global__ void get_direction(float3* directions, const float* ray_length_invs, const float3 x, const float3 y, const float3 z,
                              const float focal_length, const int width, const int height)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height)
        return;
    const int h = index(i, height - 1 - j, width);
    directions[h] = (z * focal_length + y * ((j - height / 2.f) / height) + x * ((i - width / 2.f) / height)) * ray_length_invs[h];
}
__global__ void march_ray(const float3* directions, unsigned char* pixel_values, float3 camera,
                          const float3 light, const float3 cols, const int width, const int height, const int iterations, const optimized_fractal_info params)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height)
        return;
    const int h = index(i, j, width);
    int step_num = 0;
    float min_dist = minimum_distance / (iterations * iterations + 1);
    float dist = min_dist + 1;
    float total_dist = 0;
    float3 direction = directions[h];
    while (total_dist < maximum_distance && dist > min_dist && (step_num < max_steps || dist < min_dist * 3))
    {
        dist = scaled_de(camera, iterations, side, params);
        camera = camera + direction * dist;
        total_dist += dist;
        ++step_num;
    }
    if (fabsf(dist) <= min_dist)
    {
        float brightness = 0;
        float3 off = light - camera;
        const float light_vector_length = !off;
        off = off * light_vector_length;
        float diffuse_calculated = 0;
        const float normal_angle = off & normal(camera, iterations, side, params, min_dist);
        if (normal_angle > 0)
            diffuse_calculated = fmaxf(cols.y * soft_shadow(camera, off, shadow_strength, iterations, side, params, min_dist,
                1/light_vector_length, 0.01f) * normal_angle, 0.f);
        brightness += diffuse_calculated + cols.x / (1 + step_num * ambient_occ_strength);
        brightness = fminf(fmaxf(brightness, 0.f), 1.f);
        const float col = orbit(camera, iterations, side, params);
        pixel_values[h * 3] = static_cast<unsigned char>(blue(col) * brightness * 255);
        pixel_values[h * 3 + 1] = static_cast<unsigned char>(green(col) * brightness * 255);
        pixel_values[h * 3 + 2] = static_cast<unsigned char>(red(col) * brightness * 255);
    }
    else
    {
        pixel_values[h * 3] = 0;
        pixel_values[h * 3 + 1] = 0;
        pixel_values[h * 3 + 2] = 0;
    }
}