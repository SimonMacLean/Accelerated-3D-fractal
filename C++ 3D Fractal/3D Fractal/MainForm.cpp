#include "MainForm.h"
#include <sstream>
#include <iostream>
#include <algorithm>
#include <msclr/marshal_cppstd.h>
#include "fractal_params.h"
using namespace std;
using namespace System;
using namespace Drawing;
using namespace Imaging;
using namespace Reflection;
using namespace Runtime::InteropServices;
using namespace Windows::Forms;

cudaError_t load_device_memory(float3** dev_ray_directions, float** dev_ray_lengths, unsigned char** dev_pixel_values, int draw_width,
                               int draw_height)
{
	cudaError_t cuda_status = cudaMalloc(reinterpret_cast<void**>(dev_ray_directions),
		draw_width * draw_height * sizeof(float3));
	if (cuda_status != cudaSuccess)
	{
		Console::Error->WriteLine(L"dev_ray_directions cudaMalloc failed!");
		return cuda_status;
	}
	cuda_status = cudaMalloc(reinterpret_cast<void**>(dev_ray_lengths),
		draw_width * draw_height * sizeof(float3));
	if (cuda_status != cudaSuccess)
	{
		Console::Error->WriteLine(L"dev_ray_lengths cudaMalloc failed!");
		return cuda_status;
	}
	cuda_status = cudaMalloc(reinterpret_cast<void**>(dev_pixel_values),
	                         draw_width * draw_height * 3 * sizeof(unsigned char));
	if (cuda_status != cudaSuccess)
		Console::Error->WriteLine(L"dev_pixel_values cudaMalloc failed!");
	return cuda_status;
}

cudaError_t get_direction_lengths(const dim3 grid_size, const dim3 block_size,
                                  float** dev_ray_length_invs,
                                  float focal_length, int draw_width, int draw_height)
{
	extern void get_direction_length_inv(float* ray_length_invs, const float focal_len, const int width, const int height);
	
	void* gdl_args[] = {dev_ray_length_invs, &focal_length, &draw_width, &draw_height};
	
	// Launch a kernel on the GPU with one thread for each element.
	cudaError_t cuda_status = cudaLaunchKernel((const void*)get_direction_length_inv, grid_size, block_size, gdl_args);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "get_direction_length launch failed: %s\n", cudaGetErrorString(cuda_status));
		return cuda_status;
	}
	cuda_status = cudaDeviceSynchronize();
	if (cuda_status != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching get_direction_length!\n",
		        cuda_status);
	return cuda_status;
	
}

cudaError_t get_directions(const dim3 grid_size, const dim3 block_size, float3** dev_ray_directions,
                              float** dev_ray_length_invs, float3 camera_relative_x, float3 camera_relative_y,
                              float3 camera_relative_z, float focal_length, int draw_width, int draw_height)
{
	extern void get_direction(float3* directions, const float* ray_length_invs, const float3 x, const float3 y, const float3 z,
		const float focal_length, const int width, const int height);
	void* gd_args[] = {
		dev_ray_directions, dev_ray_length_invs, &camera_relative_x, &camera_relative_y, &camera_relative_z,
		&focal_length, &draw_width, &draw_height
	};
	cudaError_t cuda_status = cudaLaunchKernel((const void*)get_direction, grid_size, block_size, gd_args);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "get_direction launch failed: %s\n", cudaGetErrorString(cuda_status));
		return cuda_status;
	}
	cuda_status = cudaDeviceSynchronize();
	if (cuda_status != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching get_direction!\n", cuda_status);
	return cuda_status;
}

cudaError_t march_rays(const dim3 grid_size, const dim3 block_size, float3** dev_ray_directions,
                       unsigned char** dev_pixel_values,
                       float3 camera_location, float3 light_location, float3 colors, int draw_width, int draw_height, int iterations,
                       fractal_creation_info params)
{
	extern void march_ray(const float3 * directions, unsigned char* pixel_values, float3 camera,
		const float3 light, const float3 cols, const int width, const int height, const int iterations, const optimized_fractal_info p);
	optimized_fractal_info o = {
		1/params.scale, sin(params.theta), cos(params.theta), sin(params.phi), cos(params.phi), params.offset
	};
	void* args[] = {
		dev_ray_directions, dev_pixel_values, &camera_location, &light_location, &colors, &draw_width, &draw_height, &iterations, &o
	};
	cudaError_t cuda_status = cudaLaunchKernel((const void*)march_ray, grid_size, block_size, args);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "march_ray launch failed: %s\n", cudaGetErrorString(cuda_status));
		return cuda_status;
	}
	cuda_status = cudaDeviceSynchronize();
	if (cuda_status != cudaSuccess)
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching march_ray!\n",
		        cuda_status);
	return cuda_status;
}

cudaError_t copy_pixels(unsigned char* dev_pixel_values, interior_ptr<cli::array<unsigned char>^> pixels,
	int draw_width, int draw_height)
{
	pin_ptr<unsigned char> pixels_start = &(*pixels)[0];
	cudaError_t cuda_status = cudaMemcpy(pixels_start, dev_pixel_values,
		draw_width * draw_height * bytes * sizeof(unsigned char),
		cudaMemcpyDeviceToHost);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		return cuda_status;
	}
	return cuda_status;
}

void free_resources(float3* dev_ray_directions, unsigned char* dev_pixels)
{
	cudaFree(dev_ray_directions);
	cudaFree(dev_pixels);
}

float operator !(const float3 v)
{
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

float3 operator ^(const float3 v1, const float3 v2)
{
	return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

float operator &(const float3 v1, const float3 v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

float3 operator /(const float3 v, const float s)
{
	return make_float3(v.x / s, v.y / s, v.z / s);
}

float3 operator *(const float3 v, const float s)
{
	return make_float3(v.x * s, v.y * s, v.z * s);
}

float3 operator +(const float3 v1, const float3 v2)
{
	return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

float3 rotate_vec(const float3 vec, const float3 axis, const float cos, const float sin)
{
	const float d = (1 - cos) * (axis & vec);
	const float3 cross = axis ^ vec;
	return make_float3(
		d * axis.x + vec.x * cos + sin * cross.x, 
		d * axis.y + vec.y * cos + sin * cross.y,
		d * axis.z + vec.z * cos + sin * cross.z
	);
}

namespace Accelerated_3D_Fractal
{
	float* dev_ray_lengths;
	float3* dev_ray_directions;
	unsigned char* dev_pixels;
	float3 camera_location;
	float3 light_location;
	float3 camera_relative_x;
	float3 camera_relative_y;
	float3 camera_relative_z;
	constexpr float scroll_wheel_movement_size = 1.f;
	fractal_creation_info params = {1, 0, 0, {0, 0, 0}};
	constexpr float3 brightness_scalars = {0.2f, 0.8f};
	fractal_creation_info presets[25] = {
		{1.8f, -0.12f, 0.5f, {0.353333f, 0.458333f, -0.081667f}},
		{1.9073f, 2.72f, -1.16f, {0.493000f, 0.532167f, -0.449167f}},
		{2.02f, -1.57f, 1.62f, {0.551667f, -1.031667f, -0.255000f}},
		{1.65f, 0.37f, -1.023f, {0.235000f, 0.036667f, 0.128333f}},
		{1.77f, -0.22f, -0.663f, {0.346667f, 0.236667f, 0.321667f}},
		{1.66f, 1.52f, 0.19f, {0.638333f, 0.323333f, 0.181667f}},
		{1.58f, -1.45f, -2.333f, {0.258333f, 0.021667f, 0.420000f}},
		{1.87f, 3.141f, 0.02f, {0.595000f, -0.021500f, -0.491667f}},
		{1.81f, 1.44f, -2.99f, {0.484167f, -0.127500f, 0.694167f}},
		{1.93f, 1.34637f, 1.58f, {0.385000f, -0.187167f, -0.260000f}},
		{1.88f, 1.52f, -1.373f, {0.756667f, 0.210000f, -0.016667f}},
		{1.6f, -2.51f, -2.353f, {0.333333f, 0.068333f, 0.238333f}},
		{2.08f, 1.493f, 3.141f, {1.238333f, -0.993333f, 1.038333f}},
		{2.0773f, 2.906f, -1.34f, {0.206333f, 0.255500f, -0.180833}},
		{1.78f, -0.1f, -3.003f, {0.245000f, -0.283333f, 0.066667f}},
		{2.0773f, 2.906f, -1.34f, {0.206333f, 0.255500f, -0.180833f}},
		{1.8093f, 3.141f, 3.074f, {0.182317f, 0.072492f, 0.518550f}},
		{1.95f, 1.570796f, 0, {1.125000f, 0.500000f, 0.000000f}},
		{1.91f, 0.06f, -0.76f, {0.573333f, 0.115000f, 0.190000f}},
		{1.8986f, -0.4166f, 0.00683f, {0.418833f, 0.901117f, 0.418333f}},
		{2.03413f, 1.688f, -1.57798f, {0.800637f, 0.683333f, 0.231772f}},
		{1.6516888f, 0.026083898f, -0.7996324f, {0.643105f, 0.856235f, 0.153051f}},
		{1.77746f, -1.66f, 0.0707307f, {0.781117f, 0.140627f, -0.330263f}},
		{2.13f, -1.77f, -1.62f, {0.831667f, 0.508333f, 0.746667f}},
		{1, 0, 0, {0.000000f, 0.000000f, 0.000000f}}
	};
	float3 mouse_location;
	constexpr float mouse_sensitivity = 0.01f;
	bool is_mouse_down;
	float focal_length = 2;
	constexpr int pixel_size = 1;
	int iterations;
	int draw_width;
	int draw_height;
	dim3 block_size = {32, 32, 1};
	dim3 grid_size;

	inline MainForm::MainForm()
	{
		InitializeComponent();
		if (cudaSetDevice(0) != cudaSuccess)
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
	inline std::string to_string(const float3 a)
	{
		return "(" + std::to_string(a.x) + "," + std::to_string(a.y) + "," + std::to_string(a.z) + ")";
	}
	bool is_float(const string s)
	{
		istringstream iss(s);
		float f;
		iss >> noskipws >> f;
		return iss.eof() && !iss.fail();
	}
	string commaToPeriod(const string& input) {
		string result = input;
		replace(result.begin(), result.end(), ',', '.');
		return result;
	}
	Void MainForm::SetScaleControls()
	{
		ScaleText->Text = params.scale.ToString("0.000");
		ScaleSlider->Value = max(ScaleSlider->Minimum,
		                         min(ScaleSlider->Maximum, static_cast<int>(ScaleSlider->Maximum * params.scale / 2)));
	}

	Void MainForm::SetThetaControls()
	{
		ThetaText->Text = params.theta.ToString("0.00");
		ThetaSlider->Value = max(ThetaSlider->Minimum,
		                         min(ThetaSlider->Maximum, static_cast<int>(ThetaSlider->Maximum / pi * params.theta)));
	}

	Void MainForm::SetPhiControls()
	{
		PhiText->Text = params.phi.ToString("0.00");
		PhiSlider->Value = max(PhiSlider->Minimum,
		                       min(PhiSlider->Maximum, static_cast<int>(PhiSlider->Maximum / pi * params.phi)));
	}

	Void MainForm::SetOffsetControls()
	{
		OffsetXText->Text = params.offset.x.ToString("0.000");
		OffsetYText->Text = params.offset.y.ToString("0.000");
		OffsetZText->Text = params.offset.z.ToString("0.000");
	}

	inline Void MainForm::OnLoad(Object^ sender, EventArgs^ e)
	{
		DrawPanel->Width = ScreenDivider->Panel2->Width;
		DrawPanel->Height = ScreenDivider->Panel2->Height;
		draw_width = DrawPanel->ClientSize.Width / pixel_size / 4 * 4;
		draw_height = DrawPanel->ClientSize.Height / pixel_size / 4 * 4;
		pixel_values = gcnew cli::array<unsigned char>(draw_width * draw_height * bytes);
		b = gcnew Bitmap(draw_width, draw_height, draw_width * bytes,
		                 PixelFormat::Format24bppRgb, Marshal::UnsafeAddrOfPinnedArrayElement(pixel_values, 0));
		camera_location = {0, 0, -3};
		light_location = {10, 20, -30};
		load_device_memory(&dev_ray_directions, &dev_ray_lengths, &dev_pixels, draw_width, draw_height);
		grid_size = {(draw_width + block_size.x - 1) / block_size.x, (draw_height + block_size.y - 1) / block_size.y};
		get_direction_lengths(grid_size, block_size, &dev_ray_lengths, focal_length, draw_width, draw_height);
		camera_relative_x = { 1, 0, 0 };
		camera_relative_y = { 0, 1, 0 };
		camera_relative_z = { 0, 0, 1 };
		get_directions(grid_size, block_size, &dev_ray_directions, &dev_ray_lengths, camera_relative_x, camera_relative_y, camera_relative_z, focal_length, draw_width, draw_height);
		DrawPanel->Invalidate();
	}

	Void MainForm::OnClosed(System::Object^ sender, System::Windows::Forms::FormClosedEventArgs^ e)
	{
		free_resources(dev_ray_directions, dev_pixels);
	}

	inline Void MainForm::PresetDropdownChanged(Object^ sender, EventArgs^ e)
	{
		params = presets[PresetDropdown->SelectedIndex];
		SetScaleControls();
		SetThetaControls();
		SetPhiControls();
		SetOffsetControls();
		iterations = IterationsSlider->Value;
		DrawPanel->Invalidate();
	}

	inline Void MainForm::OnDraw(Object^ sender, PaintEventArgs^ e)
	{
		//if (directions.Length == 0) return;
		march_rays(grid_size, block_size, &dev_ray_directions, &dev_pixels, camera_location, light_location,
		           brightness_scalars, draw_width, draw_height, iterations, params);
		copy_pixels(dev_pixels, &pixel_values, draw_width, draw_height);
		e->Graphics->DrawImage(b, 0, 0, draw_width * pixel_size, draw_height * pixel_size);
	}

	inline Void MainForm::OnMouseMove(Object^ sender, MouseEventArgs^ e)
	{
		if (is_mouse_down)
		{
			const float3 offset = {
				(e->X - mouse_location.x) * mouse_sensitivity, (e->Y - mouse_location.y) * mouse_sensitivity
			};
			const float y_rot_cos = cos(offset.x);
			const float y_rot_sin = sin(offset.x);
			const float x_rot_cos = cos(offset.y);
			const float x_rot_sin = sin(offset.y);
			camera_relative_x = rotate_vec(camera_relative_x, camera_relative_y, y_rot_cos, y_rot_sin);
			camera_relative_z = rotate_vec(camera_relative_z, camera_relative_y, y_rot_cos, y_rot_sin);
			camera_location = rotate_vec(camera_location, camera_relative_y, y_rot_cos, y_rot_sin);
			camera_relative_y = rotate_vec(camera_relative_y, camera_relative_x, x_rot_cos, x_rot_sin);
			camera_relative_z = rotate_vec(camera_relative_z, camera_relative_x, x_rot_cos, x_rot_sin);
			camera_location = rotate_vec(camera_location, camera_relative_x, x_rot_cos, x_rot_sin);
			camera_relative_x = camera_relative_x / !camera_relative_x;
			camera_relative_y = camera_relative_y / !camera_relative_y;
			camera_relative_z = camera_relative_z / !camera_relative_z;
			get_directions(grid_size, block_size, &dev_ray_directions, &dev_ray_lengths, camera_relative_x, camera_relative_y, camera_relative_z, focal_length, draw_width, draw_height);
			DrawPanel->Invalidate();
		}
		mouse_location = {static_cast<float>(e->X), static_cast<float>(e->Y)};
	}

	inline Void MainForm::OnMouseDown(Object^ sender, MouseEventArgs^ e)
	{
		mouse_location = {static_cast<float>(e->X), static_cast<float>(e->Y)};
		is_mouse_down = true;
	}

	inline Void MainForm::OnMouseUp(Object^ sender, MouseEventArgs^ e)
	{
		is_mouse_down = false;
	}

	inline Void MainForm::OnMouseWheel(Object^ sender, MouseEventArgs^ e)
	{
		if (mouse_location.x < 0 || mouse_location.y < 0 || mouse_location.x >= draw_width * pixel_size ||
			mouse_location.y >= draw_height * pixel_size)
			return;
		float3 mouse_diff = camera_relative_z * focal_length + camera_relative_x * ((mouse_location.x - draw_width / 2)/draw_width)+
			camera_relative_y * ((draw_height / 2 - mouse_location.y)/draw_width);
		mouse_diff = mouse_diff / !mouse_diff * (scroll_wheel_movement_size * e->Delta / 100.f);
		camera_location = camera_location + mouse_diff;
		DrawPanel->Invalidate();
	}

	inline Void MainForm::ScaleTextChanged(Object^ sender, EventArgs^ e)
	{
		const string s = commaToPeriod(msclr::interop::marshal_as<std::string>(ScaleText->Text));
		if (!is_float(s))
		{
			SetScaleControls();
			return;
		}
		params.scale = stof(s);
		SetScaleControls();
		DrawPanel->Invalidate();
	}

	inline Void MainForm::ScaleSliderScroll(Object^ sender, EventArgs^ e)
	{
		params.scale = ScaleSlider->Value * 2.f / ScaleSlider->Maximum;
		SetScaleControls();
		DrawPanel->Invalidate();
	}

	inline Void MainForm::ThetaTextChanged(Object^ sender, EventArgs^ e)
	{
		const string s = commaToPeriod(msclr::interop::marshal_as<std::string>(ThetaText->Text));
		if (!is_float(s))
		{
			SetThetaControls();
			return;
		}
		params.theta = stof(s);
		SetThetaControls();
		DrawPanel->Invalidate();
	}

	inline Void MainForm::ThetaSliderScroll(Object^ sender, EventArgs^ e)
	{
		params.theta = static_cast<float>(ThetaSlider->Value * pi / ThetaSlider->Maximum);
		SetThetaControls();
		DrawPanel->Invalidate();
	}

	inline Void MainForm::PhiTextChanged(Object^ sender, EventArgs^ e)
	{
		const string s = commaToPeriod(msclr::interop::marshal_as<std::string>(PhiText->Text));
		if (!is_float(s))
		{
			SetPhiControls();
			return;
		}
		params.phi = stof(s);
		SetPhiControls();
		DrawPanel->Invalidate();
	}

	inline Void MainForm::PhiSliderScroll(Object^ sender, EventArgs^ e)
	{
		params.phi = static_cast<float>(PhiSlider->Value * pi / PhiSlider->Maximum);
		SetPhiControls();
		DrawPanel->Invalidate();
	}

	inline Void MainForm::OffsetXTextChanged(Object^ sender, EventArgs^ e)
	{
		const string s = commaToPeriod(msclr::interop::marshal_as<std::string>(OffsetXText->Text));
		if (!is_float(s))
		{
			SetOffsetControls();
			return;
		}
		params.offset.x = stof(s);
		SetOffsetControls();
		DrawPanel->Invalidate();
	}

	inline Void MainForm::OffsetYTextChanged(Object^ sender, EventArgs^ e)
	{
		const string s = commaToPeriod(msclr::interop::marshal_as<std::string>(OffsetYText->Text));
		if (!is_float(s))
		{
			SetOffsetControls();
			return;
		}
		params.offset.y = stof(s);
		SetOffsetControls();
		DrawPanel->Invalidate();
	}

	inline Void MainForm::OffsetZTextChanged(Object^ sender, EventArgs^ e)
	{
		const string s = commaToPeriod(msclr::interop::marshal_as<std::string>(OffsetZText->Text));
		if (!is_float(s))
		{
			SetOffsetControls();
			return;
		}
		params.offset.z = stof(s);
		SetOffsetControls();
		DrawPanel->Invalidate();
	}

	inline Void MainForm::IterationsSliderScroll(Object^ sender, EventArgs^ e)
	{
		iterations = IterationsSlider->Value;
		DrawPanel->Invalidate();
	}
}
