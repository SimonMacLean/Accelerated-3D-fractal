#pragma once

#include <gtkmm.h>
#include "FractalDrawingArea.h"
#include "FractalParams.h"

class MainWindow : public Gtk::Window {
public:
    MainWindow();
    virtual ~MainWindow();

protected:
    // Signal handlers
    void on_preset_changed();
    void on_scale_changed();
    void on_theta_changed();
    void on_phi_changed();
    void on_offset_x_changed();
    void on_offset_y_changed();
    void on_offset_z_changed();
    void on_iterations_changed();
    
    void update_fractal_display();
    void set_preset_values(FractalPreset preset);

    // Child widgets
    Gtk::Paned m_paned;
    
    // Control panel
    Gtk::Box m_control_box;
    
    // Preset controls
    Gtk::Label m_preset_label;
    Gtk::ComboBoxText m_preset_combo;
    
    // Scale controls
    Gtk::Label m_scale_label;
    Gtk::Entry m_scale_entry;
    Gtk::Scale m_scale_slider;
    Gtk::Label m_scale_min_label, m_scale_max_label;
    
    // Theta controls
    Gtk::Label m_theta_label;
    Gtk::Entry m_theta_entry;
    Gtk::Scale m_theta_slider;
    Gtk::Label m_theta_min_label, m_theta_max_label;
    
    // Phi controls
    Gtk::Label m_phi_label;
    Gtk::Entry m_phi_entry;
    Gtk::Scale m_phi_slider;
    Gtk::Label m_phi_min_label, m_phi_max_label;
    
    // Offset controls
    Gtk::Label m_offset_label;
    Gtk::Label m_offset_x_label, m_offset_y_label, m_offset_z_label;
    Gtk::Entry m_offset_x_entry, m_offset_y_entry, m_offset_z_entry;
    
    // Iterations controls
    Gtk::Label m_iterations_label;
    Gtk::Scale m_iterations_slider;
    Gtk::Label m_iterations_min_label, m_iterations_max_label;
    
    // Drawing area
    FractalDrawingArea m_drawing_area;
    
    // Current fractal parameters
    FractalCreationInfo m_fractal_info;
    int m_iterations;
    
    // Helper methods
    void setup_ui();
    void setup_controls();
    void connect_signals();
    void update_controls_from_params();
};