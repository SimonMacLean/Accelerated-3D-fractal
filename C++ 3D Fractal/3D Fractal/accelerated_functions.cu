#include "device_launch_parameters.h"
#include "fractal_params.h"
#include <algorithm>
#include "math_functions.h"
using namespace std;
/**
 * \param v A float3 representing a vector in R³
 * \return The length of v
 */
__device__ inline float operator !(const float3 v)
{
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}
/**
 * \brief Dot product operator
 * \param v1 A float3 representing a vector in R³
 * \param v2 Another float3 representing another vector in R³
 * \return The dot product of v1 and v2
 */
__device__ inline float operator &(const float3 v1, const float3 v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}
/**
 * \brief Cross product operator
 * \param v1 A float3 representing a vector in R³
 * \param v2 Another float3 representing another vector in R³
 * \return The cross product of v1 and v2
 */
__device__ inline float3 operator ^(const float3 v1, const float3 v2)
{
    return { v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z , v1.x * v2.y - v1.y * v2.x };
}
/**
 * \brief Component-wise division
 * \param v1 A float3 representing a vector in R³
 * \param v2 Another float3 representing another vector in R³
 * \return A float3 where each component of v1 has been scaled by the inverse of the corresponding component of v2
 */
__device__ inline float3 operator /(float3 v1, const float3 v2)
{
    v1.x /= v2.x;
    v1.y /= v2.y;
    v1.z /= v2.z;
    return v1;
    //return { v1.x / v2.x,v1.y / v2.y,v1.z / v2.z };
}
/**
 * \brief Scalar division
 * \param v A float3 representing a vector in R³
 * \param s A scalar value
 * \return A float3 where each component of v has been divided by s
 */
__device__ inline float3 operator /(float3 v, const float s)
{
    v.x /= s;
    v.y /= s;
    v.z /= s;
    return v;
    //return { v.x / s,v.y / s,v.z / s };
}
/**
 * \brief Component-wise multiplication
 * \param v1 A float3 representing a vector in R³
 * \param v2 Another float3 representing another vector in R³
 * \return A float3 where each component of v1 has been scaled by the corresponding component of v2
 */
__device__ inline float3 operator *(float3 v1, const float3 v2)
{
    v1.x *= v2.x;
    v1.y *= v2.y;
    v1.z *= v2.z;
    return v1;
    //return { v1.x * v2.x,v1.y * v2.y,v1.z * v2.z };
}
/**
 * \brief Scalar multiplication
 * \param v A float3 representing a vector in R³
 * \param s A scalar value
 * \return A float3 where each component of v has been scaled by s
 */
__device__ inline float3 operator *(float3 v, const float s)
{
    v.x *= s;
    v.y *= s;
    v.z *= s;
    return v;
    //return { v.x * s,v.y * s,v.z * s };
}
/**
 * \brief Vector addition
 * \param v1 A float3 representing a vector in R³
 * \param v2 Another float3 representing another vector in R³
 * \return The sum of the two vectors
 */
__device__ inline float3 operator +(float3 v1, const float3 v2)
{
    v1.x += v2.x;
    v1.y += v2.y;
    v1.z += v2.z;
    return v1;
    //return { v1.x + v2.x,v1.y + v2.y,v1.z + v2.z };
}
/**
 * \brief Component-wise addition
 * \param v A float3 representing a vector in R³
 * \param s A scalar value
 * \return A float3 where s has been added to each component of v
 */
__device__ inline float3 operator +(float3 v, const float s)
{
    v.x += s;
    v.y += s;
    v.z += s;
    return v;
    //return { v.x + s,v.y + s,v.z + s };
}
/**
 * \brief Vector subtraction
 * \param v1 A float3 representing a vector in R³
 * \param v2 Another float3 representing another vector in R³
 * \return The signed difference between v1 and v2
 */
__device__ inline float3 operator -(float3 v1, const float3 v2)
{
    v1.x -= v2.x;
    v1.y -= v2.y;
    v1.z -= v2.z;
    return v1;
    //return { v1.x - v2.x,v1.y - v2.y,v1.z - v2.z };
}
/**
 * \brief Component-wise subtraction
 * \param v A float3 representing a vector in R³
 * \param s A scalar value
 * \return A float3 where s has been subtracted from each component of v
 */
__device__ inline float3 operator -(float3 v, const float s)
{
    v.x -= s;
    v.y -= s;
    v.z -= s;
    return v;
    //return { v.x - s,v.y - s,v.z - s };
}
/**
 * \brief Component-wise modulo
 * \param v1 A float3 representing a vector in R³
 * \param v2 Another float3 representing another vector in R³
 * \return A float3 where each component is the corresponding component of v1 modulo the corresponding component of v2
 */
__device__ inline float3 operator %(float3 v1, const float3 v2)
{
    v1.x -= static_cast<int>(v1.x / v2.x) * v2.x;
    v1.y -= static_cast<int>(v1.y / v2.y) * v2.y;
    v1.z -= static_cast<int>(v1.z / v2.z) * v2.z;
    return v1;
}
/**
 * \brief 2D to 1D array indexing
 * \param x The x component of the 2D array index
 * \param y The y component of the 2D array index
 * \param width The width of the 2D array
 * \return The index of the corresponding element in the corresponding 1D array
 */
__device__ inline int index(const int x, const int y, const int width)
{
    return x + y * width;
}
/**
 * \brief Rotates a vector about an axis by an angle
 * \param vec The vector to be rotated
 * \param axis The axis of rotation
 * \param cos The cosine of the rotation angle
 * \param sin The sine of the rotation angle
 * \return The resulting vector after vec has been rotated about axis by the desired angle
 */
__device__ inline float3 rotate_vec(const float3 vec, const float3 axis, const float cos, const float sin)
{
    const float d = (1 - cos) * (axis & vec);
    //const float3 cross = axis ^ vec;
    return { d * axis.x + vec.x * cos + sin * (axis.y * vec.z - axis.z * vec.y), d * axis.y + vec.y * cos + sin * (axis.z * vec.x - axis.x * vec.z), d * axis.z + vec.z * cos + sin * (axis.x * vec.y - axis.y * vec.x) };
}
/**
 * \brief Scales a scalar value by the inverse of another a specified number of times
 * \param s The value to be scaled
 * \param iterations The number of times to scale s
 * \param scale The value that s will be divided by
 * \return s*scale^-iterations
 */
__device__ inline float warp_scalar(float s, const int iterations, const float scale)
{
    for (int i = 0; i < iterations; ++i)
        s /= scale;
    return s;
}
__device__ inline float cube_distance_equation(const float3 p, const float3 cube_center, const float cube_side_length)
{
    return max(abs(p.x - cube_center.x), max(abs(p.y - cube_center.y), abs(p.z - cube_center.z))) - cube_side_length / 2;
}
__device__ inline float distance_equation(const float3 p, const float side)
{
    return max(abs(p.x), max(abs(p.y), abs(p.z))) - side / 2;
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
    float a = min(vec.x - vec.y, 0.f);
    vec.x -= a;
    vec.y += a;
    a = min(vec.x - vec.z, 0.f);
    vec.x -= a;
    vec.z += a;
    a = min(vec.y - vec.z, 0.f);
    vec.y -= a;
    vec.z += a;
    return vec;
}
__device__ inline float3 max_space(float3 a, const float3 b)
{
    a.x = max(a.x, b.x);
    a.y = max(a.y, b.y);
	a.z = max(a.z, b.z);
    return a;
}
__device__ inline float3 max_space(float3 a, const float b)
{
    a.x = max(a.x, b);
    a.y = max(a.y, b);
    a.z = max(a.z, b);
    return a;
}
__device__ inline float3 min_space(float3 a, const float3 b)
{
    a.x = min(a.x, b.x);
    a.y = min(a.y, b.y);
    a.z = min(a.z, b.z);
    return a;
}
__device__ inline float3 min_space(float3 a, const float b)
{
    a.x = min(a.x, b);
    a.y = min(a.y, b);
    a.z = min(a.z, b);
    return a;
}
__device__ inline float3 fold_box(const float3 z, const float r)
{
    return max_space(min_space(z, r), -r) * 2 - z;
}
__device__ inline float3 abs_space(float3 p)
{
    p.x = abs(p.x);
    p.y = abs(p.y);
    p.z = abs(p.z);
    return p;
}
__device__ inline float3 abs_space_x(float3 p)
{
    p.x = abs(p.x);
    return p;
}
__device__ inline float3 abs_space_y(float3 p)
{
    p.y = abs(p.y);
    return p;
}
__device__ inline float3 abs_space_z(float3 p)
{
    p.z = abs(p.z);
    return p;
}
__device__ inline float3 rotate_x(const float3 z, const float s, const float c)
{
    return{ z.x, c * z.y + s * z.z,c * z.z - s * z.y };
}
__device__ inline float3 rotate_y(const float3 z, const float s, const float c)
{
    return { c * z.x - s * z.z, z.y,c * z.z + s * z.x };
}
__device__ inline float3 rotate_z(const float3 z, const float s, const float c)
{
    return { c * z.x + s * z.y,c * z.y - s * z.x, z.z };
}
__device__ inline float3 transform(float3 p, optimized_fractal_info params)
{
    p = p * params.scale;
    p = abs_space(p);
    p = rotate_z(p,params.sin_theta, params.cos_theta);
    p = fold_menger(p);
    p = rotate_x(p, params.sin_phi, params.cos_phi);
    p = p - params.offset;
    return p;
}
__device__ inline float3 warp_space(float3 p, const int iterations, optimized_fractal_info params)
{
    for (int i = 0; i < iterations; ++i)
        p = transform(p, params);
    return p;
}
__device__ inline float scaled_de(const float3 p, const int iterations, const float side, optimized_fractal_info params)
{
    return warp_scalar(distance_equation(warp_space(p, iterations, params), side), iterations, params.scale);
}
__device__ inline float3 normal(const float3 p, const int iterations, const float side, optimized_fractal_info params, float epsilon)
{
    const float3 scaled = {
        scaled_de({ p.x + epsilon, p.y, p.z }, iterations, side, params) -
        scaled_de({ p.x - epsilon, p.y, p.z }, iterations, side, params),
        scaled_de({ p.x, p.y + epsilon, p.z }, iterations, side, params) -
        scaled_de({ p.x, p.y - epsilon, p.z }, iterations, side, params),
        scaled_de({ p.x, p.y, p.z + epsilon }, iterations, side, params) -
        scaled_de({ p.x, p.y, p.z - epsilon }, iterations, side, params)
    };
    return scaled / !scaled;
}
__device__ inline float soft_shadow(const float3 p, const float3 d, const float shadow_strength, const int iterations, const float side, optimized_fractal_info params, const float minDist, const float maxDist, const float minAngle)
{
    float darkness = 1;
    float prev_dist = 2147483647;
    float angle = 1;
    float total_dist = minDist;
    while (total_dist < maxDist)
    {
        const float dist = scaled_de(p + d * total_dist, iterations, side, params);
        const float old_new_int_dist = dist * dist / (2 * prev_dist);
        const float leg_length = sqrtf(dist * dist - old_new_int_dist * old_new_int_dist);
        angle = shadow_strength * leg_length / max(0.f, total_dist - old_new_int_dist);
        darkness = min(darkness, angle);
        prev_dist = dist;
        total_dist += dist;
        if (dist < 0 || darkness < minAngle)
            return 0;
    }
    return darkness;
}
__device__ inline float orbit(float3 p, const int iterations, const float side, optimized_fractal_info params)
{
    const float direction = scaled_de(p, 1, side, params);
    for (int i = 0; i < iterations; ++i)
    {
        p = transform(p, params);
        const float f = warp_scalar(distance_equation(p, side), i, params.scale);
        if (f * direction >= 0)
            return f * params.scale * 6;
    }
    return warp_scalar(distance_equation(p, side), iterations - 1, params.scale) * 6;
}
__device__ inline float trapezoid_wave(const float loc)
{
    return min(max(abs(loc - 3), 0.f) - 1, 1.f);
}
__device__ inline float red(const float loc)
{
    return trapezoid_wave(loc - static_cast<int>(loc / 6) * 6);
}
__device__ inline float green(const float loc)
{
    return trapezoid_wave(loc + 4 - static_cast<int>((loc + 4) / 6) * 6);
}
__device__ inline float blue(const float loc)
{
    return trapezoid_wave(loc + 2 - static_cast<int>((loc + 2) / 6) * 6);
}
__global__ void get_direction(float3* directions, float focal_length, int width, int height)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int h = index(i, height - 1 - j, width);
    const float3 p = { -1 * focal_length,(j - height / 2.f) / height,(i - width / 2.f) / height };
    directions[h] = p / !p;
}
__global__ void rotate_direction(float3* directions, float3 axis, float cos, float sin, int width, int height)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int h = index(i, height - 1 - j, width);
    directions[h] = rotate_vec(directions[h], axis, cos, sin);
    directions[h] = directions[h] / !directions[h];
}
__global__ void march_ray(float3* directions, unsigned char* pixel_values, float3 camera,
    float3 light, float3 cols, int width, int iterations, optimized_fractal_info params)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int h = index(i, j, width);
    int step_num = 0;
    float dist = minimum_distance + 1;
    float total_dist = 0;
    while (total_dist < maximum_distance && dist > minimum_distance && step_num < max_steps)
    {
        dist = scaled_de(camera, iterations, side, params);
        camera = camera + directions[h] * dist;
        total_dist += dist;
        ++step_num;
    }
    if (abs(dist) <= minimum_distance)
    {
        float brightness = 0;
        float3 off = light - camera;
        const float light_vector_length = !off;
        off = off / light_vector_length;
        float diffuse_calculated = 0;
        const float normal_angle = off & normal(camera, iterations, side, params, minimum_distance);
        if (normal_angle > 0)
            diffuse_calculated = max(cols.y * soft_shadow(camera, off, shadow_strength, iterations, side, params, minimum_distance,
                light_vector_length, 0.01f) * normal_angle, 0.f);
        brightness += diffuse_calculated + cols.x / (1 + step_num * ambient_occ_strength);
        brightness = min(max(brightness, 0.f), 1.f);
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