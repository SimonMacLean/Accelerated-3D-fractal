#pragma once

#include <gtkmm.h>
#include "FractalEngine.h"

class FractalDrawingArea : public Gtk::DrawingArea {
public:
    FractalDrawingArea();
    virtual ~FractalDrawingArea();
    
    // Update fractal parameters and trigger redraw
    void update_fractal(const OptimizedFractalInfo& params, int iterations);
    
    // Set camera and light positions
    void set_camera_position(const Vec3& pos);
    void set_light_position(const Vec3& pos);
    
protected:
    // Override default signal handlers
    bool on_draw(const Cairo::RefPtr<Cairo::Context>& cr) override;
    bool on_button_press_event(GdkEventButton* button_event) override;
    bool on_button_release_event(GdkEventButton* button_event) override;
    bool on_motion_notify_event(GdkEventMotion* motion_event) override;
    bool on_scroll_event(GdkEventScroll* scroll_event) override;
    
private:
    FractalEngine m_engine;
    OptimizedFractalInfo m_fractal_params;
    int m_iterations;
    
    Vec3 m_camera_pos;
    Vec3 m_light_pos;
    
    // Mouse interaction state
    bool m_mouse_pressed;
    double m_last_mouse_x, m_last_mouse_y;
    
    // Pixel buffer for rendering
    std::vector<unsigned char> m_pixel_buffer;
    
    // Render dimensions
    int m_render_width, m_render_height;
    
    void initialize_fractal();
    void render_fractal();
};