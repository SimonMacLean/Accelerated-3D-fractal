# GTK 3D Fractal

A GTK-based 3D fractal viewer implemented in C++ using gtkmm.

## Features

- Real-time 3D fractal rendering using CPU ray marching
- Interactive parameter controls (scale, theta, phi, offset, iterations)
- Multiple fractal presets
- Mouse navigation (pan with left-click drag, zoom with scroll wheel)
- Cross-platform GTK user interface

## Building

### Prerequisites

- CMake (>= 3.16)
- gtkmm-3.0 development packages
- C++17 compatible compiler

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install libgtkmm-3.0-dev pkg-config cmake g++
```

### Building the Project
```bash
cd "GTK 3D Fractal"
mkdir build
cd build
cmake ..
make
```

### Running
```bash
./GTK_3D_Fractal
```

## Controls

- **Preset**: Select from predefined fractal configurations
- **Scale**: Adjust the fractal scaling factor (0.0 - 2.0)
- **θ (Theta)**: Rotation angle around Z-axis (-π to π)
- **φ (Phi)**: Rotation angle around X-axis (-π to π)
- **Offset**: X, Y, Z translation values
- **Iterations**: Number of fractal iterations (0-20)

## Mouse Interaction

- **Left-click + drag**: Pan the camera
- **Scroll wheel**: Zoom in/out

## Technical Details

This application ports the fractal algorithms from the original CUDA-accelerated version to a CPU-based implementation suitable for cross-platform GTK applications. The core fractal mathematics including Menger folding, space transformations, and ray marching are preserved from the original implementation.

The rendering uses distance estimation and ray marching techniques with soft shadows and orbit trap coloring to create visually appealing 3D fractal structures.