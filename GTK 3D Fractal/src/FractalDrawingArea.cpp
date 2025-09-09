#include "FractalDrawingArea.h"
#include <iostream>

FractalDrawingArea::FractalDrawingArea() 
    : m_iterations(5), 
      m_camera_pos(0.0f, 0.0f, -3.0f),
      m_light_pos(2.0f, 2.0f, -1.0f),
      m_mouse_pressed(false),
      m_last_mouse_x(0), 
      m_last_mouse_y(0),
      m_render_width(640),
      m_render_height(480) {
    
    // Enable events
    add_events(Gdk::BUTTON_PRESS_MASK | 
               Gdk::BUTTON_RELEASE_MASK | 
               Gdk::POINTER_MOTION_MASK |
               Gdk::SCROLL_MASK);
    
    initialize_fractal();
}

FractalDrawingArea::~FractalDrawingArea() {
}

void FractalDrawingArea::initialize_fractal() {
    // Initialize with default parameters
    FractalCreationInfo creation_info;
    creation_info.scale = 1.0f;
    creation_info.theta = 0.0f;
    creation_info.phi = 0.0f;
    creation_info.offset = Vec3(0.0f, 0.0f, 0.0f);
    
    m_fractal_params = m_engine.optimizeParams(creation_info);
    render_fractal();
}

void FractalDrawingArea::update_fractal(const OptimizedFractalInfo& params, int iterations) {
    m_fractal_params = params;
    m_iterations = iterations;
    render_fractal();
}

void FractalDrawingArea::set_camera_position(const Vec3& pos) {
    m_camera_pos = pos;
    render_fractal();
}

void FractalDrawingArea::set_light_position(const Vec3& pos) {
    m_light_pos = pos;
    render_fractal();
}

void FractalDrawingArea::render_fractal() {
    // Get the current allocation
    auto allocation = get_allocation();
    m_render_width = std::max(allocation.get_width(), 1);
    m_render_height = std::max(allocation.get_height(), 1);
    
    // Render the fractal
    m_engine.render(m_pixel_buffer, m_render_width, m_render_height,
                   m_fractal_params, m_iterations, m_camera_pos, m_light_pos);
    
    // Queue a redraw
    queue_draw();
}

bool FractalDrawingArea::on_draw(const Cairo::RefPtr<Cairo::Context>& cr) {
    if (m_pixel_buffer.empty()) {
        return false;
    }
    
    // Create Cairo surface from pixel buffer
    auto surface = Cairo::ImageSurface::create(Cairo::FORMAT_RGB24, m_render_width, m_render_height);
    
    // Get surface data
    unsigned char* surface_data = surface->get_data();
    int stride = surface->get_stride();
    
    // Copy pixel data to Cairo surface (convert RGB to BGRA)
    for (int y = 0; y < m_render_height; ++y) {
        for (int x = 0; x < m_render_width; ++x) {
            int src_index = (y * m_render_width + x) * 3;
            int dst_index = y * stride + x * 4;
            
            if (src_index + 2 < static_cast<int>(m_pixel_buffer.size())) {
                surface_data[dst_index + 0] = m_pixel_buffer[src_index + 2]; // Blue
                surface_data[dst_index + 1] = m_pixel_buffer[src_index + 1]; // Green
                surface_data[dst_index + 2] = m_pixel_buffer[src_index + 0]; // Red
                surface_data[dst_index + 3] = 255; // Alpha
            }
        }
    }
    
    surface->mark_dirty();
    
    // Draw the surface
    cr->set_source(surface, 0, 0);
    cr->paint();
    
    return true;
}

bool FractalDrawingArea::on_button_press_event(GdkEventButton* button_event) {
    if (button_event->button == 1) { // Left mouse button
        m_mouse_pressed = true;
        m_last_mouse_x = button_event->x;
        m_last_mouse_y = button_event->y;
        return true;
    }
    return false;
}

bool FractalDrawingArea::on_button_release_event(GdkEventButton* button_event) {
    if (button_event->button == 1) { // Left mouse button
        m_mouse_pressed = false;
        return true;
    }
    return false;
}

bool FractalDrawingArea::on_motion_notify_event(GdkEventMotion* motion_event) {
    if (m_mouse_pressed) {
        double dx = motion_event->x - m_last_mouse_x;
        double dy = motion_event->y - m_last_mouse_y;
        
        // Rotate camera based on mouse movement
        float rotation_speed = 0.01f;
        
        // Simple camera rotation (this could be enhanced)
        m_camera_pos.x += dx * rotation_speed;
        m_camera_pos.y -= dy * rotation_speed;
        
        m_last_mouse_x = motion_event->x;
        m_last_mouse_y = motion_event->y;
        
        render_fractal();
        return true;
    }
    return false;
}

bool FractalDrawingArea::on_scroll_event(GdkEventScroll* scroll_event) {
    float zoom_speed = 0.1f;
    
    if (scroll_event->direction == GDK_SCROLL_UP) {
        // Zoom in
        m_camera_pos.z += zoom_speed;
    } else if (scroll_event->direction == GDK_SCROLL_DOWN) {
        // Zoom out
        m_camera_pos.z -= zoom_speed;
    }
    
    render_fractal();
    return true;
}