
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <string>
#include <cmath>
#include <algorithm>
#include <limits>
using namespace std;
__device__ float operator !(const float3 p)
{
    return sqrtf(p.x * p.x + p.y * p.y + p.z * p.z);
}
__device__ float operator &(const float3 a, const float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ float3 operator ^(const float3 a, const float3 b)
{
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z , a.x * b.y - a.y * b.x };
}
__device__ float3 operator /(const float3 a, const float3 b)
{
    return { a.x / b.x,a.y / b.y,a.z / b.z };
}
__device__ float3 operator /(const float3 a, const float b)
{
    return { a.x / b,a.y / b,a.z / b };
}
__device__ float3 operator *(const float3 a, const float3 b)
{
    return { a.x * b.x,a.y * b.y,a.z * b.z };

}
__device__ float3 operator *(const float3 a, const float b)
{
    return { a.x * b,a.y * b,a.z * b };
}
__device__ float3 operator +(const float3 a, const float3 b)
{
    return { a.x + b.x,a.y + b.y,a.z + b.z };
}
__device__ float3 operator +(const float3 a, const float b)
{
    return { a.x + b,a.y + b,a.z + b };
}
__device__ float3 operator -(const float3 a, const float3 b)
{
    return { a.x - b.x,a.y - b.y,a.z - b.z };
}
__device__ float3 operator -(const float3 a, const float b)
{
    return { a.x - b,a.y - b,a.z - b };
}
__device__ float3 operator %(const float3 a, const float3 b)
{
    return { a.x - static_cast<int>(a.x / b.x) * b.x,a.y - static_cast<int>(a.y / b.y) * b.y,a.z - static_cast<int>(a.z / b.z) * b.z };
}
__device__ int index(const int x, const int y, const int width)
{
    return y * width + x;
}
__device__ float3 rotate_vec(const float3 p, const float3 a, const float t, const float c, const float s)
{
    const float d = t * (a & p);
    const float3 x = a ^ p;
    return { d * a.x + p.x * c + s * x.x, d * a.y + p.y * c + s * x.y, d * a.z + p.z * c + s * x.z };
}
/*__global__ void GetColor(Color* colors)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    colors[i] = Color.FromArgb(i, i, i);
}*/
__device__ float warp_dist(float d, const int iterations, const float scale)
{
    for (int i = 0; i < iterations; ++i)
        d /= scale;
    return d;
}
__device__ float cube_de(const float3 p, const float3 c, const float di)
{
    const float3 o = p - c;
    return max(abs(o.x), max(abs(o.y), abs(o.z))) - di / 2;
}
__device__ float de(const float3 p, const float side)
{
    return max(abs(p.x), max(abs(p.y), abs(p.z))) - side / 2;
}
__device__ float3 mod_space(const float3 p, const float3 mod)
{
    return ((p + mod / 2) % mod + mod) % mod - mod / 2;
}
__device__ float3 fold_space(const float3 p, const float3 n)
{
    if ((p & n) >= 0)
        return p;
    return p - (n * 2 * (p & n) / (n & n));
}
__device__ float3 fold_menger(float3 z)
{
    float a = min(z.x - z.y, 0.f);
    z.x -= a;
    z.y += a;
    a = min(z.x - z.z, 0.f);
    z.x -= a;
    z.z += a;
    a = min(z.y - z.z, 0.f);
    z.y -= a;
    z.z += a;
    return z;
}
__device__ float3 max_space(const float3 a, const float3 b)
{
    return { max(a.x, b.x),
    max(a.y, b.y),
    max(a.z, b.z) };
}
__device__ float3 max_space(const float3 a, const float b)
{
    return { max(a.x, b),
    max(a.y, b),
    max(a.z, b) };
}
__device__ float3 min_space(const float3 a, const float3 b)
{
    return { min(a.x, b.x),
    min(a.y, b.y),
    min(a.z, b.z) };
}
__device__ float3 min_space(const float3 a, const float b)
{
    return { min(a.x, b),
    min(a.y, b),
    min(a.z, b) };
}
__device__ float3 fold_box(const float3 z, const float r)
{
    return max_space(min_space(z, r), -r) * 2 - z;
}
__device__ float3 abs_space(const float3 p)
{
    return { abs(p.x),abs(p.y),abs(p.z) };
}
__device__ float3 abs_space_x(const float3 p)
{
    return{ abs(p.x),p.y,p.z };
}
__device__ float3 abs_space_y(const float3 p)
{
    return{ p.x,abs(p.y),p.z };
}
__device__ float3 abs_space_z(const float3 p)
{
    return{ p.x,p.y,abs(p.z) };
}
__device__ float3 rotate_x(const float3 z, const float t)
{
    const float s = sin(t);
    const float c = cos(t);
    return{ z.x, c * z.y + s * z.z,c * z.z - s * z.y };
}
__device__ float3 rotate_y(const float3 z, const float t)
{
    const float s = sin(t);
    const float c = cos(t);
    return { c * z.x - s * z.z, z.y,c * z.z + s * z.x };
}
__device__ float3 rotate_z(const float3 z, const float t)
{
    const float s = sin(t);
    const float c = cos(t);
    return { c * z.x + s * z.y,c * z.y - s * z.x, z.z };
}
__device__ float3 transform(float3 p, int iterations, const float3 seed, const float3 shift)
{
    p = p * seed.x;
    p = abs_space(p);
    p = rotate_z(p, seed.y);
    p = fold_menger(p);
    p = rotate_x(p, seed.z);
    p = p - shift;
    return p;
}
__device__ float3 warp_space(float3 p, const int iterations, const float3 seed, const float3 shift)
{
    for (int i = 0; i < iterations; ++i)
        p = transform(p, iterations, seed, shift);
    return p;
}
__device__ float scaled_de(const float3 p, const int iterations, const float side, const float3 seed, const float3 shift)
{
    return warp_dist(de(warp_space(p, iterations, seed, shift), side), iterations, seed.x);
}
__device__ float3 normal(const float3 p, const int iterations, const float side, const float3 seed, const float3 shift, const float epsilon)
{
    const float3 scaled = {
        scaled_de({ p.x + epsilon, p.y, p.z }, iterations, side, seed, shift) -
        scaled_de({ p.x - epsilon, p.y, p.z }, iterations, side, seed, shift),
        scaled_de({ p.x, p.y + epsilon, p.z }, iterations, side, seed, shift) -
        scaled_de({ p.x, p.y - epsilon, p.z }, iterations, side, seed, shift),
        scaled_de({ p.x, p.y, p.z + epsilon }, iterations, side, seed, shift) -
        scaled_de({ p.x, p.y, p.z - epsilon }, iterations, side, seed, shift)
    };
    return scaled / !scaled;
}
__device__ float new_soft_shadow(const float3 p, const float3 d, const float shadow_strength, const int iterations, const float side, const float3 seed, const float3 shift, const float minDist, const float maxDist, const float minAngle)
{
    float darkness = 1;
    float prev_dist = 2147483647;
    float angle = 1;
    float total_dist = minDist;
    while (total_dist < maxDist)
    {
        const float dist = scaled_de(p + d * total_dist, iterations, side, seed, shift);
        const float old_new_int_dist = dist * dist / (2 * prev_dist);
        const float leg_length = sqrt(dist * dist - old_new_int_dist * old_new_int_dist);
        angle = shadow_strength * leg_length / max(0.f, total_dist - old_new_int_dist);
        darkness = min(darkness, angle);
        prev_dist = dist;
        total_dist += dist;
        if (dist < 0 || darkness < minAngle)
            return 0;
    }
    return darkness;
}
__device__ float orbit(float3 p, const int iterations, const float side, const float3 seed, const float3 shift)
{
    const float direction = scaled_de(p, 1, side, seed, shift);
    for (int i = 0; i < iterations; ++i)
    {
        p = transform(p, iterations, seed, shift);
        if (warp_dist(de(p, side), i, seed.x) * direction >= 0)
            return warp_dist(de(p, side), i - 1, seed.x) * 6;
    }
    return warp_dist(de(p, side), iterations - 1, seed.x) * 6;
}
__device__ float trapezoid_wave(const float loc)
{
    return min(max(abs(loc - 3), 0.f) - 1, 1.f);
}
__device__ float red(const float loc)
{
    return trapezoid_wave(loc - static_cast<int>(loc / 6) * 6);
}
__device__ float green(const float loc)
{
    return trapezoid_wave(loc + 4 - static_cast<int>((loc + 4) / 6) * 6);
}
__device__ float blue(const float loc)
{
    return trapezoid_wave(loc + 2 - static_cast<int>((loc + 2) / 6) * 6);
}
__global__ void get_direction(float3* directions, const float focal_length, const int width, const int height)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int h = index(i, height - 1 - j, width);
    const float3 p = { focal_length,(j - height / 2.f) / height,(i - width / 2.f) / height };
    directions[h] = p / !p;
}
__global__ void rotate_direction(float3* directions, const float3 a, const float t, const float c, const float s, const int width, const int height)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int h = index(i, height - 1 - j, width);
    directions[h] = rotate_vec(directions[h], a, t, c, s);
    directions[h] = directions[h] / !directions[h];
}
__global__ void march_ray(const float3* directions, unsigned char* pixel_values, const float3 camera,
    const float3 light, const float2 cols, const float min_dist, const float max_dist, const int max_step, int bytes, const int width, const int iterations,
    const float side, const float3 seed, const float3 shift, const float shadow_strength, const float ambient_occ_strength)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int h = index(i, j, width);
    float3 p = camera;
    int step_num = 0;
    float dist = min_dist + 1;
    float total_dist = 0;
    while (total_dist < max_dist && dist > min_dist && step_num != max_step)
    {
        dist = scaled_de(p, iterations, side, seed, shift);
        p = p + directions[h] * dist;
        total_dist += dist;
        ++step_num;
    }
    if (abs(dist) <= min_dist)
    {
        float brightness = 0;
        float3 off = light - p;
        const float light_vector_length = !off;
        off = off / light_vector_length;
        float diffuse_calculated = 0;
        const float normal_angle = (off & normal(p, iterations, side, seed, shift, min_dist));
        if (normal_angle > 0)
            diffuse_calculated = max(cols.y * new_soft_shadow(p, off, shadow_strength, iterations, side, seed, shift, min_dist,
                light_vector_length, 0.01f) * normal_angle, 0.f);
        brightness += diffuse_calculated + cols.x / (1 + step_num * ambient_occ_strength);
        brightness = min(max(brightness, 0.f), 1.f);
        const float col = orbit(p, iterations, side, seed, shift);
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
cudaError_t add_with_cuda(float3* c, const float3* a, const float3* b, unsigned int size);
__global__ void add(float3* c, const float3* a, const float3* b)
{
    const int i = threadIdx.x;
    c[i] = { a[i].x + b[i].x,a[i].y + b[i].y,a[i].z + b[i].z };
}
string to_string(const float3 a)
{
    return "(" + to_string(a.x) + "," + to_string(a.y) + "," + to_string(a.z) + ")";
}
int main()
{
    constexpr int array_size = 5;
    const float3 a[array_size] = { {1, 2, 3}, make_float3(4, 5, 6), make_float3(7, 8, 9), make_float3(10, 11, 12), make_float3(13, 14, 15) };
    const float3 b[array_size] = { make_float3(10, 20, 30), make_float3(40, 50, 60), make_float3(70, 80, 90), make_float3(100, 110, 120), make_float3(130, 140, 150) };
    float3 c[array_size] = { {0,0,0} };

    // Add vectors in parallel.
    cudaError_t cuda_status = add_with_cuda(c, a, b, array_size);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("a + b = {%s, %s, %s, %s, %s}\n",
        to_string(c[0]).c_str(), to_string(c[1]).c_str(), to_string(c[2]).c_str(), to_string(c[3]).c_str(), to_string(c[4]).c_str());

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cuda_status = cudaDeviceReset();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
// Helper function for using CUDA to add vectors in parallel.
cudaError_t add_with_cuda(float3* c, const float3* a, const float3* b, unsigned int size)
{
    float3* dev_a = nullptr;
    float3* dev_b = nullptr;
    float3* dev_c = nullptr;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaError_t cuda_status = cudaSetDevice(0);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cuda_status = cudaMalloc(reinterpret_cast<void**>(&dev_c), size * sizeof(float3));
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cuda_status = cudaMalloc(reinterpret_cast<void**>(&dev_a), size * sizeof(float3));
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cuda_status = cudaMalloc(reinterpret_cast<void**>(&dev_b), size * sizeof(float3));
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cuda_status = cudaMemcpy(dev_a, a, size * sizeof(float3), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cuda_status = cudaMemcpy(dev_b, b, size * sizeof(float3), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    add <<<1, size >>> (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cuda_status));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cuda_status);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cuda_status = cudaMemcpy(c, dev_c, size * sizeof(float3), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cuda_status;
}