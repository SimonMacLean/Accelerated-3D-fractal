#include <gtkmm.h>
#include "MainWindow.h"
#include <iostream>

int main(int argc, char* argv[]) {
    try {
        auto app = Gtk::Application::create(argc, argv, "org.example.gtk3dfractal");
        
        MainWindow window;
        
        return app->run(window);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}