#include "MainWindow.h"
#include <iostream>

MainWindow::MainWindow() 
    : m_control_box(Gtk::ORIENTATION_VERTICAL),
      m_iterations(5) {
    
    set_title("GTK 3D Fractal");
    set_default_size(1200, 800);
    
    // Initialize fractal parameters
    m_fractal_info.scale = 1.0f;
    m_fractal_info.theta = 0.0f;
    m_fractal_info.phi = 0.0f;
    m_fractal_info.offset = Vec3(0.0f, 0.0f, 0.0f);
    
    setup_ui();
    connect_signals();
    update_fractal_display();
}

MainWindow::~MainWindow() {
}

void MainWindow::setup_ui() {
    // Add main paned container
    add(m_paned);
    
    // Set up control panel (left side)
    m_paned.pack1(m_control_box, false, false);
    m_control_box.set_size_request(300, -1);
    m_control_box.set_margin_left(10);
    m_control_box.set_margin_right(10);
    m_control_box.set_margin_top(10);
    m_control_box.set_margin_bottom(10);
    
    setup_controls();
    
    // Set up drawing area (right side)
    m_paned.pack2(m_drawing_area, true, false);
    
    show_all_children();
}

void MainWindow::setup_controls() {
    // Preset controls
    m_preset_label.set_text("Preset:");
    m_preset_label.set_halign(Gtk::ALIGN_START);
    m_control_box.pack_start(m_preset_label, Gtk::PACK_SHRINK);
    
    for (const auto& name : PRESET_NAMES) {
        m_preset_combo.append(name);
    }
    m_preset_combo.set_active(0); // Custom
    m_control_box.pack_start(m_preset_combo, Gtk::PACK_SHRINK);
    
    // Add some spacing
    auto spacer1 = Gtk::manage(new Gtk::Label(" "));
    m_control_box.pack_start(*spacer1, Gtk::PACK_SHRINK);
    
    // Scale controls
    m_scale_label.set_text("Scale:");
    m_scale_label.set_halign(Gtk::ALIGN_START);
    m_control_box.pack_start(m_scale_label, Gtk::PACK_SHRINK);
    
    m_scale_entry.set_text("1.000");
    m_scale_entry.set_width_chars(8);
    m_control_box.pack_start(m_scale_entry, Gtk::PACK_SHRINK);
    
    m_scale_slider.set_range(0.0, 2.0);
    m_scale_slider.set_value(1.0);
    m_scale_slider.set_digits(3);
    m_control_box.pack_start(m_scale_slider, Gtk::PACK_SHRINK);
    
    auto scale_box = Gtk::manage(new Gtk::Box(Gtk::ORIENTATION_HORIZONTAL));
    m_scale_min_label.set_text("0.000");
    m_scale_max_label.set_text("2.000");
    m_scale_min_label.set_halign(Gtk::ALIGN_START);
    m_scale_max_label.set_halign(Gtk::ALIGN_END);
    scale_box->pack_start(m_scale_min_label, Gtk::PACK_SHRINK);
    scale_box->pack_end(m_scale_max_label, Gtk::PACK_SHRINK);
    m_control_box.pack_start(*scale_box, Gtk::PACK_SHRINK);
    
    // Add spacing
    auto spacer2 = Gtk::manage(new Gtk::Label(" "));
    m_control_box.pack_start(*spacer2, Gtk::PACK_SHRINK);
    
    // Theta controls
    m_theta_label.set_text("θ:");
    m_theta_label.set_halign(Gtk::ALIGN_START);
    m_control_box.pack_start(m_theta_label, Gtk::PACK_SHRINK);
    
    m_theta_entry.set_text("0.00");
    m_theta_entry.set_width_chars(8);
    m_control_box.pack_start(m_theta_entry, Gtk::PACK_SHRINK);
    
    m_theta_slider.set_range(-PI, PI);
    m_theta_slider.set_value(0.0);
    m_theta_slider.set_digits(2);
    m_control_box.pack_start(m_theta_slider, Gtk::PACK_SHRINK);
    
    auto theta_box = Gtk::manage(new Gtk::Box(Gtk::ORIENTATION_HORIZONTAL));
    m_theta_min_label.set_text("-π");
    m_theta_max_label.set_text("π");
    m_theta_min_label.set_halign(Gtk::ALIGN_START);
    m_theta_max_label.set_halign(Gtk::ALIGN_END);
    theta_box->pack_start(m_theta_min_label, Gtk::PACK_SHRINK);
    theta_box->pack_end(m_theta_max_label, Gtk::PACK_SHRINK);
    m_control_box.pack_start(*theta_box, Gtk::PACK_SHRINK);
    
    // Add spacing
    auto spacer3 = Gtk::manage(new Gtk::Label(" "));
    m_control_box.pack_start(*spacer3, Gtk::PACK_SHRINK);
    
    // Phi controls
    m_phi_label.set_text("φ:");
    m_phi_label.set_halign(Gtk::ALIGN_START);
    m_control_box.pack_start(m_phi_label, Gtk::PACK_SHRINK);
    
    m_phi_entry.set_text("0.00");
    m_phi_entry.set_width_chars(8);
    m_control_box.pack_start(m_phi_entry, Gtk::PACK_SHRINK);
    
    m_phi_slider.set_range(-PI, PI);
    m_phi_slider.set_value(0.0);
    m_phi_slider.set_digits(2);
    m_control_box.pack_start(m_phi_slider, Gtk::PACK_SHRINK);
    
    auto phi_box = Gtk::manage(new Gtk::Box(Gtk::ORIENTATION_HORIZONTAL));
    m_phi_min_label.set_text("-π");
    m_phi_max_label.set_text("π");
    m_phi_min_label.set_halign(Gtk::ALIGN_START);
    m_phi_max_label.set_halign(Gtk::ALIGN_END);
    phi_box->pack_start(m_phi_min_label, Gtk::PACK_SHRINK);
    phi_box->pack_end(m_phi_max_label, Gtk::PACK_SHRINK);
    m_control_box.pack_start(*phi_box, Gtk::PACK_SHRINK);
    
    // Add spacing
    auto spacer4 = Gtk::manage(new Gtk::Label(" "));
    m_control_box.pack_start(*spacer4, Gtk::PACK_SHRINK);
    
    // Offset controls
    m_offset_label.set_text("Offset:");
    m_offset_label.set_halign(Gtk::ALIGN_START);
    m_control_box.pack_start(m_offset_label, Gtk::PACK_SHRINK);
    
    auto offset_x_box = Gtk::manage(new Gtk::Box(Gtk::ORIENTATION_HORIZONTAL));
    m_offset_x_label.set_text("X:");
    m_offset_x_entry.set_text("0.00");
    m_offset_x_entry.set_width_chars(6);
    offset_x_box->pack_start(m_offset_x_label, Gtk::PACK_SHRINK);
    offset_x_box->pack_start(m_offset_x_entry, Gtk::PACK_SHRINK);
    m_control_box.pack_start(*offset_x_box, Gtk::PACK_SHRINK);
    
    auto offset_y_box = Gtk::manage(new Gtk::Box(Gtk::ORIENTATION_HORIZONTAL));
    m_offset_y_label.set_text("Y:");
    m_offset_y_entry.set_text("0.00");
    m_offset_y_entry.set_width_chars(6);
    offset_y_box->pack_start(m_offset_y_label, Gtk::PACK_SHRINK);
    offset_y_box->pack_start(m_offset_y_entry, Gtk::PACK_SHRINK);
    m_control_box.pack_start(*offset_y_box, Gtk::PACK_SHRINK);
    
    auto offset_z_box = Gtk::manage(new Gtk::Box(Gtk::ORIENTATION_HORIZONTAL));
    m_offset_z_label.set_text("Z:");
    m_offset_z_entry.set_text("0.00");
    m_offset_z_entry.set_width_chars(6);
    offset_z_box->pack_start(m_offset_z_label, Gtk::PACK_SHRINK);
    offset_z_box->pack_start(m_offset_z_entry, Gtk::PACK_SHRINK);
    m_control_box.pack_start(*offset_z_box, Gtk::PACK_SHRINK);
    
    // Add spacing
    auto spacer5 = Gtk::manage(new Gtk::Label(" "));
    m_control_box.pack_start(*spacer5, Gtk::PACK_SHRINK);
    
    // Iterations controls
    m_iterations_label.set_text("Iterations:");
    m_iterations_label.set_halign(Gtk::ALIGN_START);
    m_control_box.pack_start(m_iterations_label, Gtk::PACK_SHRINK);
    
    m_iterations_slider.set_range(0, 20);
    m_iterations_slider.set_value(5);
    m_iterations_slider.set_digits(0);
    m_control_box.pack_start(m_iterations_slider, Gtk::PACK_SHRINK);
    
    auto iterations_box = Gtk::manage(new Gtk::Box(Gtk::ORIENTATION_HORIZONTAL));
    m_iterations_min_label.set_text("0");
    m_iterations_max_label.set_text("20");
    m_iterations_min_label.set_halign(Gtk::ALIGN_START);
    m_iterations_max_label.set_halign(Gtk::ALIGN_END);
    iterations_box->pack_start(m_iterations_min_label, Gtk::PACK_SHRINK);
    iterations_box->pack_end(m_iterations_max_label, Gtk::PACK_SHRINK);
    m_control_box.pack_start(*iterations_box, Gtk::PACK_SHRINK);
}

void MainWindow::connect_signals() {
    m_preset_combo.signal_changed().connect(sigc::mem_fun(*this, &MainWindow::on_preset_changed));
    m_scale_entry.signal_changed().connect(sigc::mem_fun(*this, &MainWindow::on_scale_changed));
    m_scale_slider.signal_value_changed().connect(sigc::mem_fun(*this, &MainWindow::on_scale_changed));
    m_theta_entry.signal_changed().connect(sigc::mem_fun(*this, &MainWindow::on_theta_changed));
    m_theta_slider.signal_value_changed().connect(sigc::mem_fun(*this, &MainWindow::on_theta_changed));
    m_phi_entry.signal_changed().connect(sigc::mem_fun(*this, &MainWindow::on_phi_changed));
    m_phi_slider.signal_value_changed().connect(sigc::mem_fun(*this, &MainWindow::on_phi_changed));
    m_offset_x_entry.signal_changed().connect(sigc::mem_fun(*this, &MainWindow::on_offset_x_changed));
    m_offset_y_entry.signal_changed().connect(sigc::mem_fun(*this, &MainWindow::on_offset_y_changed));
    m_offset_z_entry.signal_changed().connect(sigc::mem_fun(*this, &MainWindow::on_offset_z_changed));
    m_iterations_slider.signal_value_changed().connect(sigc::mem_fun(*this, &MainWindow::on_iterations_changed));
}

void MainWindow::on_preset_changed() {
    int active = m_preset_combo.get_active_row_number();
    if (active > 0) {
        set_preset_values(static_cast<FractalPreset>(active));
        update_controls_from_params();
        update_fractal_display();
    }
}

void MainWindow::on_scale_changed() {
    try {
        double value = std::stod(m_scale_entry.get_text());
        m_fractal_info.scale = static_cast<float>(value);
        m_scale_slider.set_value(value);
        update_fractal_display();
    } catch (const std::exception&) {
        // Invalid input, ignore
    }
}

void MainWindow::on_theta_changed() {
    try {
        double value = std::stod(m_theta_entry.get_text());
        m_fractal_info.theta = static_cast<float>(value);
        m_theta_slider.set_value(value);
        update_fractal_display();
    } catch (const std::exception&) {
        // Invalid input, ignore
    }
}

void MainWindow::on_phi_changed() {
    try {
        double value = std::stod(m_phi_entry.get_text());
        m_fractal_info.phi = static_cast<float>(value);
        m_phi_slider.set_value(value);
        update_fractal_display();
    } catch (const std::exception&) {
        // Invalid input, ignore
    }
}

void MainWindow::on_offset_x_changed() {
    try {
        double value = std::stod(m_offset_x_entry.get_text());
        m_fractal_info.offset.x = static_cast<float>(value);
        update_fractal_display();
    } catch (const std::exception&) {
        // Invalid input, ignore
    }
}

void MainWindow::on_offset_y_changed() {
    try {
        double value = std::stod(m_offset_y_entry.get_text());
        m_fractal_info.offset.y = static_cast<float>(value);
        update_fractal_display();
    } catch (const std::exception&) {
        // Invalid input, ignore
    }
}

void MainWindow::on_offset_z_changed() {
    try {
        double value = std::stod(m_offset_z_entry.get_text());
        m_fractal_info.offset.z = static_cast<float>(value);
        update_fractal_display();
    } catch (const std::exception&) {
        // Invalid input, ignore
    }
}

void MainWindow::on_iterations_changed() {
    m_iterations = static_cast<int>(m_iterations_slider.get_value());
    update_fractal_display();
}

void MainWindow::update_fractal_display() {
    FractalEngine engine;
    OptimizedFractalInfo optimized = engine.optimizeParams(m_fractal_info);
    m_drawing_area.update_fractal(optimized, m_iterations);
}

void MainWindow::set_preset_values(FractalPreset preset) {
    FractalEngine engine;
    m_fractal_info = engine.getPresetConfig(preset);
}

void MainWindow::update_controls_from_params() {
    m_scale_entry.set_text(std::to_string(m_fractal_info.scale));
    m_scale_slider.set_value(m_fractal_info.scale);
    
    m_theta_entry.set_text(std::to_string(m_fractal_info.theta));
    m_theta_slider.set_value(m_fractal_info.theta);
    
    m_phi_entry.set_text(std::to_string(m_fractal_info.phi));
    m_phi_slider.set_value(m_fractal_info.phi);
    
    m_offset_x_entry.set_text(std::to_string(m_fractal_info.offset.x));
    m_offset_y_entry.set_text(std::to_string(m_fractal_info.offset.y));
    m_offset_z_entry.set_text(std::to_string(m_fractal_info.offset.z));
}