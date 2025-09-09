#pragma once

#include <array>
#include <cmath>

constexpr double PI = 3.14159265358979323846;
constexpr float MINIMUM_DISTANCE = 0.0002f;
constexpr float MAXIMUM_DISTANCE = 30.0f;
constexpr float AMBIENT_OCC_STRENGTH = 0.05f;
constexpr float SHADOW_STRENGTH = 8.0f;
constexpr float SIDE = 1.0f;
constexpr int MAX_STEPS = 512;

struct Vec3 {
    float x, y, z;
    
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    
    Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }
    
    Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }
    
    Vec3 operator*(float s) const {
        return Vec3(x * s, y * s, z * s);
    }
    
    Vec3 operator/(float s) const {
        return Vec3(x / s, y / s, z / s);
    }
    
    Vec3& operator+=(const Vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
    
    Vec3& operator-=(const Vec3& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }
    
    float dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    
    float length() const {
        return std::sqrt(x * x + y * y + z * z);
    }
    
    Vec3 normalized() const {
        float len = length();
        if (len == 0) return Vec3();
        return *this / len;
    }
};

struct FractalCreationInfo {
    float scale;
    float theta;
    float phi;
    Vec3 offset;
};

struct OptimizedFractalInfo {
    float scale;
    float sin_theta;
    float cos_theta;
    float sin_phi;
    float cos_phi;
    Vec3 offset;
};

// Preset fractal configurations
enum class FractalPreset {
    CUSTOM = 0,
    JUMP_THE_CRATER,
    TOO_MANY_TREES,
    HOLE_IN_ONE,
    AROUND_THE_WORLD,
    THE_HILLS_ARE_ALIVE,
    BEWARE_OF_BUMPS,
    MOUNTAIN_CLIMBING,
    THE_CATWALK,
    MIND_THE_GAP,
    DONT_GET_CRUSHED,
    THE_SPONGE,
    RIDE_THE_GECKO,
    BUILD_UP_SPEED,
    AROUND_THE_CITADEL,
    PLANET_CRUSHER,
    TOP_OF_THE_CITADEL,
    BUILDING_BRIDGES,
    PYLON_PALACE,
    THE_CROWN_JEWELS,
    EXPRESSWAYS,
    BUNNY_HOPS,
    ASTEROID_FIELD,
    LILY_PADS,
    FATAL_FISSURES
};

const std::array<const char*, 25> PRESET_NAMES = {
    "Custom...",
    "Jump the Crater",
    "Too Many Trees", 
    "Hole in One",
    "Around the World",
    "The Hills Are Alive",
    "Beware of Bumps",
    "Mountain Climbing",
    "The Catwalk",
    "Mind the Gap",
    "Don't Get Crushed",
    "The Sponge",
    "Ride the Gecko",
    "Build Up Speed",
    "Around the Citadel",
    "Planet Crusher",
    "Top of the Citadel",
    "Building Bridges",
    "Pylon Palace",
    "The Crown Jewels",
    "Expressways",
    "Bunny Hops",
    "Asteroid Field",
    "Lily Pads",
    "Fatal Fissures"
};