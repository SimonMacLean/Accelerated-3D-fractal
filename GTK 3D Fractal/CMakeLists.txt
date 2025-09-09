cmake_minimum_required(VERSION 3.16)
project(GTK_3D_Fractal)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTKMM gtkmm-3.0 REQUIRED)

# Include directories
include_directories(${GTKMM_INCLUDE_DIRS})

# Link directories
link_directories(${GTKMM_LIBRARY_DIRS})

# Add compile flags
add_definitions(${GTKMM_CFLAGS_OTHER})

# Source files
set(SOURCES
    src/main.cpp
    src/MainWindow.cpp
    src/FractalDrawingArea.cpp
    src/FractalEngine.cpp
)

set(HEADERS
    include/MainWindow.h
    include/FractalDrawingArea.h
    include/FractalEngine.h
    include/FractalParams.h
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(${PROJECT_NAME} ${GTKMM_LIBRARIES})

# Include directories for our headers
target_include_directories(${PROJECT_NAME} PRIVATE include)

# Set compiler flags
target_compile_options(${PROJECT_NAME} PRIVATE ${GTKMM_CFLAGS_OTHER})