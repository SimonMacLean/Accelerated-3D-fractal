#include <cstdio>
#include <cuda_runtime.h>

#include "MainForm.h"
using namespace System;
using namespace Windows::Forms;

[STAThread]
int main(array<String^>^ args) {

    Application::EnableVisualStyles();
    Application::SetCompatibleTextRenderingDefault(false);
    Application::Run(gcnew Accelerated_3D_Fractal::MainForm());
	if (cudaDeviceReset() != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
    return 0;
}
