#pragma once
namespace Accelerated_3D_Fractal {
	using namespace System;
	using namespace ComponentModel;
	using namespace Collections;
	using namespace Windows::Forms;
	using namespace Data;
	using namespace Drawing;
	/// <summary>
	/// The main WinForm
	/// </summary>
	public ref class MainForm : public System::Windows::Forms::Form
	{
	public:
		MainForm();

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~MainForm()
		{
			if (components)
				delete components;
		}
	private:
		System::Windows::Forms::SplitContainer^ ScreenDivider;
		System::Windows::Forms::Label^ ThetaMaxLabel;
		System::Windows::Forms::Label^ ThetaMinLabel;
		System::Windows::Forms::TrackBar^ ThetaSlider;
		System::Windows::Forms::TextBox^ ThetaText;
		System::Windows::Forms::Label^ ThetaLabel;
		System::Windows::Forms::Label^ ScaleMaxLabel;
		System::Windows::Forms::Label^ ScaleMinLabel;
		System::Windows::Forms::TrackBar^ ScaleSlider;
		System::Windows::Forms::TextBox^ ScaleText;
		System::Windows::Forms::Label^ ScaleLabel;
		System::Windows::Forms::ComboBox^ PresetDropdown;
		System::Windows::Forms::Label^ PresetLabel;
		System::Windows::Forms::Label^ IterationsMaxLabel;
		System::Windows::Forms::Label^ IterationsMinLabel;
		System::Windows::Forms::TrackBar^ IterationsSlider;
		System::Windows::Forms::Label^ IterationsLabel;
		System::Windows::Forms::TextBox^ OffsetZText;
		System::Windows::Forms::Label^ OffsetZLabel;
		System::Windows::Forms::TextBox^ OffsetYText;
		System::Windows::Forms::Label^ OffsetYLabel;
		System::Windows::Forms::TextBox^ OffsetXText;
		System::Windows::Forms::Label^ OffsetXLabel;
		System::Windows::Forms::Label^ OffsetLabel;
		System::Windows::Forms::Label^ PhiMaxLabel;
		System::Windows::Forms::Label^ PhiMinLabel;
		System::Windows::Forms::TrackBar^ PhiSlider;
		System::Windows::Forms::TextBox^ PhiText;
		System::Windows::Forms::Label^ PhiLabel;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		System::ComponentModel::Container^ components;
#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent()
		{
			this->ScreenDivider = (gcnew System::Windows::Forms::SplitContainer());
			this->IterationsMaxLabel = (gcnew System::Windows::Forms::Label());
			this->IterationsMinLabel = (gcnew System::Windows::Forms::Label());
			this->IterationsSlider = (gcnew System::Windows::Forms::TrackBar());
			this->IterationsLabel = (gcnew System::Windows::Forms::Label());
			this->OffsetZText = (gcnew System::Windows::Forms::TextBox());
			this->OffsetZLabel = (gcnew System::Windows::Forms::Label());
			this->OffsetYText = (gcnew System::Windows::Forms::TextBox());
			this->OffsetYLabel = (gcnew System::Windows::Forms::Label());
			this->OffsetXText = (gcnew System::Windows::Forms::TextBox());
			this->OffsetXLabel = (gcnew System::Windows::Forms::Label());
			this->OffsetLabel = (gcnew System::Windows::Forms::Label());
			this->PhiMaxLabel = (gcnew System::Windows::Forms::Label());
			this->PhiMinLabel = (gcnew System::Windows::Forms::Label());
			this->PhiSlider = (gcnew System::Windows::Forms::TrackBar());
			this->PhiText = (gcnew System::Windows::Forms::TextBox());
			this->PhiLabel = (gcnew System::Windows::Forms::Label());
			this->ThetaMaxLabel = (gcnew System::Windows::Forms::Label());
			this->ThetaMinLabel = (gcnew System::Windows::Forms::Label());
			this->ThetaSlider = (gcnew System::Windows::Forms::TrackBar());
			this->ThetaText = (gcnew System::Windows::Forms::TextBox());
			this->ThetaLabel = (gcnew System::Windows::Forms::Label());
			this->ScaleMaxLabel = (gcnew System::Windows::Forms::Label());
			this->ScaleMinLabel = (gcnew System::Windows::Forms::Label());
			this->ScaleSlider = (gcnew System::Windows::Forms::TrackBar());
			this->ScaleText = (gcnew System::Windows::Forms::TextBox());
			this->ScaleLabel = (gcnew System::Windows::Forms::Label());
			this->PresetDropdown = (gcnew System::Windows::Forms::ComboBox());
			this->PresetLabel = (gcnew System::Windows::Forms::Label());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->ScreenDivider))->BeginInit();
			this->ScreenDivider->Panel1->SuspendLayout();
			this->ScreenDivider->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->IterationsSlider))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->PhiSlider))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->ThetaSlider))->BeginInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->ScaleSlider))->BeginInit();
			this->SuspendLayout();
			// 
			// ScreenDivider
			// 
			this->ScreenDivider->Dock = System::Windows::Forms::DockStyle::Fill;
			this->ScreenDivider->Location = System::Drawing::Point(0, 0);
			this->ScreenDivider->Name = L"ScreenDivider";
			// 
			// ScreenDivider.Panel1
			// 
			this->ScreenDivider->Panel1->Controls->Add(this->IterationsMaxLabel);
			this->ScreenDivider->Panel1->Controls->Add(this->IterationsMinLabel);
			this->ScreenDivider->Panel1->Controls->Add(this->IterationsSlider);
			this->ScreenDivider->Panel1->Controls->Add(this->IterationsLabel);
			this->ScreenDivider->Panel1->Controls->Add(this->OffsetZText);
			this->ScreenDivider->Panel1->Controls->Add(this->OffsetZLabel);
			this->ScreenDivider->Panel1->Controls->Add(this->OffsetYText);
			this->ScreenDivider->Panel1->Controls->Add(this->OffsetYLabel);
			this->ScreenDivider->Panel1->Controls->Add(this->OffsetXText);
			this->ScreenDivider->Panel1->Controls->Add(this->OffsetXLabel);
			this->ScreenDivider->Panel1->Controls->Add(this->OffsetLabel);
			this->ScreenDivider->Panel1->Controls->Add(this->PhiMaxLabel);
			this->ScreenDivider->Panel1->Controls->Add(this->PhiMinLabel);
			this->ScreenDivider->Panel1->Controls->Add(this->PhiSlider);
			this->ScreenDivider->Panel1->Controls->Add(this->PhiText);
			this->ScreenDivider->Panel1->Controls->Add(this->PhiLabel);
			this->ScreenDivider->Panel1->Controls->Add(this->ThetaMaxLabel);
			this->ScreenDivider->Panel1->Controls->Add(this->ThetaMinLabel);
			this->ScreenDivider->Panel1->Controls->Add(this->ThetaSlider);
			this->ScreenDivider->Panel1->Controls->Add(this->ThetaText);
			this->ScreenDivider->Panel1->Controls->Add(this->ThetaLabel);
			this->ScreenDivider->Panel1->Controls->Add(this->ScaleMaxLabel);
			this->ScreenDivider->Panel1->Controls->Add(this->ScaleMinLabel);
			this->ScreenDivider->Panel1->Controls->Add(this->ScaleSlider);
			this->ScreenDivider->Panel1->Controls->Add(this->ScaleText);
			this->ScreenDivider->Panel1->Controls->Add(this->ScaleLabel);
			this->ScreenDivider->Panel1->Controls->Add(this->PresetDropdown);
			this->ScreenDivider->Panel1->Controls->Add(this->PresetLabel);
			// 
			// ScreenDivider.Panel2
			// 
			this->ScreenDivider->Panel2->Paint += gcnew System::Windows::Forms::PaintEventHandler(this, &MainForm::OnDraw);
			this->ScreenDivider->Panel2->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &MainForm::OnMouseDown);
			this->ScreenDivider->Panel2->MouseMove += gcnew System::Windows::Forms::MouseEventHandler(this, &MainForm::OnMouseMove);
			this->ScreenDivider->Panel2->MouseUp += gcnew System::Windows::Forms::MouseEventHandler(this, &MainForm::OnMouseUp);
			this->ScreenDivider->Panel2->MouseWheel += gcnew System::Windows::Forms::MouseEventHandler(this, &MainForm::OnMouseWheel);
			this->ScreenDivider->Size = System::Drawing::Size(982, 703);
			this->ScreenDivider->SplitterDistance = 331;
			this->ScreenDivider->TabIndex = 0;
			// 
			// IterationsMaxLabel
			// 
			this->IterationsMaxLabel->AutoSize = true;
			this->IterationsMaxLabel->Font = (gcnew System::Drawing::Font(L"Times New Roman", 10.8F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->IterationsMaxLabel->Location = System::Drawing::Point(294, 536);
			this->IterationsMaxLabel->Name = L"IterationsMaxLabel";
			this->IterationsMaxLabel->Size = System::Drawing::Size(27, 20);
			this->IterationsMaxLabel->TabIndex = 27;
			this->IterationsMaxLabel->Text = L"20";
			this->IterationsMaxLabel->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// IterationsMinLabel
			// 
			this->IterationsMinLabel->AutoSize = true;
			this->IterationsMinLabel->Font = (gcnew System::Drawing::Font(L"Times New Roman", 10.8F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->IterationsMinLabel->Location = System::Drawing::Point(12, 536);
			this->IterationsMinLabel->Name = L"IterationsMinLabel";
			this->IterationsMinLabel->Size = System::Drawing::Size(18, 20);
			this->IterationsMinLabel->TabIndex = 26;
			this->IterationsMinLabel->Text = L"0";
			this->IterationsMinLabel->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// IterationsSlider
			// 
			this->IterationsSlider->Location = System::Drawing::Point(3, 500);
			this->IterationsSlider->Maximum = 20;
			this->IterationsSlider->Name = L"IterationsSlider";
			this->IterationsSlider->Size = System::Drawing::Size(321, 56);
			this->IterationsSlider->TabIndex = 25;
			this->IterationsSlider->Value = 5;
			this->IterationsSlider->Scroll += gcnew System::EventHandler(this, &MainForm::IterationsSliderScroll);
			// 
			// IterationsLabel
			// 
			this->IterationsLabel->AutoSize = true;
			this->IterationsLabel->Font = (gcnew System::Drawing::Font(L"Helvetica", 10.8F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->IterationsLabel->Location = System::Drawing::Point(12, 476);
			this->IterationsLabel->Name = L"IterationsLabel";
			this->IterationsLabel->Size = System::Drawing::Size(84, 21);
			this->IterationsLabel->TabIndex = 24;
			this->IterationsLabel->Text = L"Iterations";
			this->IterationsLabel->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// OffsetZText
			// 
			this->OffsetZText->Font = (gcnew System::Drawing::Font(L"Helvetica", 10.8F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->OffsetZText->Location = System::Drawing::Point(75, 434);
			this->OffsetZText->Name = L"OffsetZText";
			this->OffsetZText->Size = System::Drawing::Size(58, 29);
			this->OffsetZText->TabIndex = 23;
			this->OffsetZText->Text = L"0.00";
			this->OffsetZText->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			this->OffsetZText->TextChanged += gcnew System::EventHandler(this, &MainForm::OffsetZTextChanged);
			// 
			// OffsetZLabel
			// 
			this->OffsetZLabel->AutoSize = true;
			this->OffsetZLabel->Font = (gcnew System::Drawing::Font(L"Helvetica", 10.8F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->OffsetZLabel->Location = System::Drawing::Point(43, 437);
			this->OffsetZLabel->Name = L"OffsetZLabel";
			this->OffsetZLabel->Size = System::Drawing::Size(26, 21);
			this->OffsetZLabel->TabIndex = 22;
			this->OffsetZLabel->Text = L"Z:";
			this->OffsetZLabel->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// OffsetYText
			// 
			this->OffsetYText->Font = (gcnew System::Drawing::Font(L"Helvetica", 10.8F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->OffsetYText->Location = System::Drawing::Point(75, 399);
			this->OffsetYText->Name = L"OffsetYText";
			this->OffsetYText->Size = System::Drawing::Size(58, 29);
			this->OffsetYText->TabIndex = 21;
			this->OffsetYText->Text = L"0.00";
			this->OffsetYText->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			this->OffsetYText->TextChanged += gcnew System::EventHandler(this, &MainForm::OffsetYTextChanged);
			// 
			// OffsetYLabel
			// 
			this->OffsetYLabel->AutoSize = true;
			this->OffsetYLabel->Font = (gcnew System::Drawing::Font(L"Helvetica", 10.8F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->OffsetYLabel->Location = System::Drawing::Point(43, 402);
			this->OffsetYLabel->Name = L"OffsetYLabel";
			this->OffsetYLabel->Size = System::Drawing::Size(26, 21);
			this->OffsetYLabel->TabIndex = 20;
			this->OffsetYLabel->Text = L"Y:";
			this->OffsetYLabel->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// OffsetXText
			// 
			this->OffsetXText->Font = (gcnew System::Drawing::Font(L"Helvetica", 10.8F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->OffsetXText->Location = System::Drawing::Point(75, 364);
			this->OffsetXText->Name = L"OffsetXText";
			this->OffsetXText->Size = System::Drawing::Size(58, 29);
			this->OffsetXText->TabIndex = 19;
			this->OffsetXText->Text = L"0.00";
			this->OffsetXText->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			this->OffsetXText->TextChanged += gcnew System::EventHandler(this, &MainForm::OffsetXTextChanged);
			// 
			// OffsetXLabel
			// 
			this->OffsetXLabel->AutoSize = true;
			this->OffsetXLabel->Font = (gcnew System::Drawing::Font(L"Helvetica", 10.8F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->OffsetXLabel->Location = System::Drawing::Point(43, 367);
			this->OffsetXLabel->Name = L"OffsetXLabel";
			this->OffsetXLabel->Size = System::Drawing::Size(27, 21);
			this->OffsetXLabel->TabIndex = 18;
			this->OffsetXLabel->Text = L"X:";
			this->OffsetXLabel->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// OffsetLabel
			// 
			this->OffsetLabel->AutoSize = true;
			this->OffsetLabel->Font = (gcnew System::Drawing::Font(L"Helvetica", 10.8F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->OffsetLabel->Location = System::Drawing::Point(12, 340);
			this->OffsetLabel->Name = L"OffsetLabel";
			this->OffsetLabel->Size = System::Drawing::Size(63, 21);
			this->OffsetLabel->TabIndex = 17;
			this->OffsetLabel->Text = L"Offset:";
			this->OffsetLabel->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// PhiMaxLabel
			// 
			this->PhiMaxLabel->AutoSize = true;
			this->PhiMaxLabel->Font = (gcnew System::Drawing::Font(L"Times New Roman", 10.8F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->PhiMaxLabel->Location = System::Drawing::Point(303, 306);
			this->PhiMaxLabel->Name = L"PhiMaxLabel";
			this->PhiMaxLabel->Size = System::Drawing::Size(18, 20);
			this->PhiMaxLabel->TabIndex = 16;
			this->PhiMaxLabel->Text = L"π";
			this->PhiMaxLabel->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// PhiMinLabel
			// 
			this->PhiMinLabel->AutoSize = true;
			this->PhiMinLabel->Font = (gcnew System::Drawing::Font(L"Times New Roman", 10.8F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->PhiMinLabel->Location = System::Drawing::Point(12, 306);
			this->PhiMinLabel->Name = L"PhiMinLabel";
			this->PhiMinLabel->Size = System::Drawing::Size(24, 20);
			this->PhiMinLabel->TabIndex = 15;
			this->PhiMinLabel->Text = L"-π";
			this->PhiMinLabel->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// PhiSlider
			// 
			this->PhiSlider->Location = System::Drawing::Point(3, 281);
			this->PhiSlider->Maximum = 1000;
			this->PhiSlider->Minimum = -1000;
			this->PhiSlider->Name = L"PhiSlider";
			this->PhiSlider->Size = System::Drawing::Size(321, 56);
			this->PhiSlider->TabIndex = 14;
			this->PhiSlider->TickStyle = System::Windows::Forms::TickStyle::None;
			this->PhiSlider->Scroll += gcnew System::EventHandler(this, &MainForm::PhiSliderScroll);
			// 
			// PhiText
			// 
			this->PhiText->Font = (gcnew System::Drawing::Font(L"Helvetica", 10.8F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->PhiText->Location = System::Drawing::Point(44, 254);
			this->PhiText->Name = L"PhiText";
			this->PhiText->Size = System::Drawing::Size(58, 29);
			this->PhiText->TabIndex = 13;
			this->PhiText->Text = L"0.00";
			this->PhiText->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			this->PhiText->TextChanged += gcnew System::EventHandler(this, &MainForm::PhiTextChanged);
			// 
			// PhiLabel
			// 
			this->PhiLabel->AutoSize = true;
			this->PhiLabel->Font = (gcnew System::Drawing::Font(L"Helvetica", 10.8F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->PhiLabel->Location = System::Drawing::Point(12, 257);
			this->PhiLabel->Name = L"PhiLabel";
			this->PhiLabel->Size = System::Drawing::Size(28, 21);
			this->PhiLabel->TabIndex = 12;
			this->PhiLabel->Text = L"φ:";
			this->PhiLabel->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// ThetaMaxLabel
			// 
			this->ThetaMaxLabel->AutoSize = true;
			this->ThetaMaxLabel->Font = (gcnew System::Drawing::Font(L"Times New Roman", 10.8F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->ThetaMaxLabel->Location = System::Drawing::Point(303, 217);
			this->ThetaMaxLabel->Name = L"ThetaMaxLabel";
			this->ThetaMaxLabel->Size = System::Drawing::Size(18, 20);
			this->ThetaMaxLabel->TabIndex = 11;
			this->ThetaMaxLabel->Text = L"π";
			this->ThetaMaxLabel->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// ThetaMinLabel
			// 
			this->ThetaMinLabel->AutoSize = true;
			this->ThetaMinLabel->Font = (gcnew System::Drawing::Font(L"Times New Roman", 10.8F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->ThetaMinLabel->Location = System::Drawing::Point(12, 217);
			this->ThetaMinLabel->Name = L"ThetaMinLabel";
			this->ThetaMinLabel->Size = System::Drawing::Size(24, 20);
			this->ThetaMinLabel->TabIndex = 10;
			this->ThetaMinLabel->Text = L"-π";
			this->ThetaMinLabel->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// ThetaSlider
			// 
			this->ThetaSlider->Location = System::Drawing::Point(3, 192);
			this->ThetaSlider->Maximum = 1000;
			this->ThetaSlider->Minimum = -1000;
			this->ThetaSlider->Name = L"ThetaSlider";
			this->ThetaSlider->Size = System::Drawing::Size(321, 56);
			this->ThetaSlider->TabIndex = 9;
			this->ThetaSlider->TickStyle = System::Windows::Forms::TickStyle::None;
			this->ThetaSlider->Scroll += gcnew System::EventHandler(this, &MainForm::ThetaSliderScroll);
			// 
			// ThetaText
			// 
			this->ThetaText->Font = (gcnew System::Drawing::Font(L"Helvetica", 10.8F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->ThetaText->Location = System::Drawing::Point(44, 165);
			this->ThetaText->Name = L"ThetaText";
			this->ThetaText->Size = System::Drawing::Size(58, 29);
			this->ThetaText->TabIndex = 8;
			this->ThetaText->Text = L"0.00";
			this->ThetaText->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			this->ThetaText->TextChanged += gcnew System::EventHandler(this, &MainForm::ThetaTextChanged);
			// 
			// ThetaLabel
			// 
			this->ThetaLabel->AutoSize = true;
			this->ThetaLabel->Font = (gcnew System::Drawing::Font(L"Helvetica", 10.8F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->ThetaLabel->Location = System::Drawing::Point(12, 168);
			this->ThetaLabel->Name = L"ThetaLabel";
			this->ThetaLabel->Size = System::Drawing::Size(26, 21);
			this->ThetaLabel->TabIndex = 7;
			this->ThetaLabel->Text = L"θ:";
			this->ThetaLabel->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// ScaleMaxLabel
			// 
			this->ScaleMaxLabel->AutoSize = true;
			this->ScaleMaxLabel->Font = (gcnew System::Drawing::Font(L"Times New Roman", 10.8F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->ScaleMaxLabel->Location = System::Drawing::Point(271, 128);
			this->ScaleMaxLabel->Name = L"ScaleMaxLabel";
			this->ScaleMaxLabel->Size = System::Drawing::Size(50, 20);
			this->ScaleMaxLabel->TabIndex = 6;
			this->ScaleMaxLabel->Text = L"2.000";
			this->ScaleMaxLabel->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// ScaleMinLabel
			// 
			this->ScaleMinLabel->AutoSize = true;
			this->ScaleMinLabel->Font = (gcnew System::Drawing::Font(L"Times New Roman", 10.8F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->ScaleMinLabel->Location = System::Drawing::Point(12, 128);
			this->ScaleMinLabel->Name = L"ScaleMinLabel";
			this->ScaleMinLabel->Size = System::Drawing::Size(50, 20);
			this->ScaleMinLabel->TabIndex = 5;
			this->ScaleMinLabel->Text = L"0.000";
			this->ScaleMinLabel->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// ScaleSlider
			// 
			this->ScaleSlider->Location = System::Drawing::Point(3, 103);
			this->ScaleSlider->Maximum = 2000;
			this->ScaleSlider->Name = L"ScaleSlider";
			this->ScaleSlider->Size = System::Drawing::Size(321, 56);
			this->ScaleSlider->TabIndex = 4;
			this->ScaleSlider->TickStyle = System::Windows::Forms::TickStyle::None;
			this->ScaleSlider->Value = 1000;
			this->ScaleSlider->Scroll += gcnew System::EventHandler(this, &MainForm::ScaleSliderScroll);
			// 
			// ScaleText
			// 
			this->ScaleText->Font = (gcnew System::Drawing::Font(L"Times New Roman", 12, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->ScaleText->Location = System::Drawing::Point(78, 76);
			this->ScaleText->Name = L"ScaleText";
			this->ScaleText->Size = System::Drawing::Size(58, 30);
			this->ScaleText->TabIndex = 3;
			this->ScaleText->Text = L"1.000";
			this->ScaleText->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			this->ScaleText->TextChanged += gcnew System::EventHandler(this, &MainForm::ScaleTextChanged);
			// 
			// ScaleLabel
			// 
			this->ScaleLabel->AutoSize = true;
			this->ScaleLabel->Font = (gcnew System::Drawing::Font(L"Helvetica", 10.8F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->ScaleLabel->Location = System::Drawing::Point(12, 79);
			this->ScaleLabel->Name = L"ScaleLabel";
			this->ScaleLabel->Size = System::Drawing::Size(60, 21);
			this->ScaleLabel->TabIndex = 2;
			this->ScaleLabel->Text = L"Scale:";
			this->ScaleLabel->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// PresetDropdown
			// 
			this->PresetDropdown->AutoCompleteCustomSource->AddRange(gcnew cli::array< System::String^  >(25) {
				L"Jump the Crater", L"Too Many Trees",
					L"Hole in One", L"Around the World", L"The Hills Are Alive", L"Beware of Bumps", L"Mountain Climbing", L"The Catwalk", L"Mind the Gap",
					L"Don’t Get Crushed", L"The Sponge", L"Ride the Gecko", L"Build Up Speed", L"Around the Citadel", L"Planet Crusher", L"Top of the Citadel",
					L"Building Bridges", L"Pylon Palace", L"The Crown Jewels", L"Expressways", L"Bunny Hops", L"Asteroid Field", L"Lily Pads", L"Fatal Fissures",
					L"Custom…"
			});
			this->PresetDropdown->AutoCompleteMode = System::Windows::Forms::AutoCompleteMode::Append;
			this->PresetDropdown->AutoCompleteSource = System::Windows::Forms::AutoCompleteSource::CustomSource;
			this->PresetDropdown->Font = (gcnew System::Drawing::Font(L"Helvetica", 10.8F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->PresetDropdown->Items->AddRange(gcnew cli::array< System::Object^  >(25) {
				L"Jump the Crater", L"Too Many Trees", L"Hole in One",
					L"Around the World", L"The Hills Are Alive", L"Beware of Bumps", L"Mountain Climbing", L"The Catwalk", L"Mind the Gap", L"Don’t Get Crushed",
					L"The Sponge", L"Ride the Gecko", L"Build Up Speed", L"Around the Citadel", L"Planet Crusher", L"Top of the Citadel", L"Building Bridges",
					L"Pylon Palace", L"The Crown Jewels", L"Expressways", L"Bunny Hops", L"Asteroid Field", L"Lily Pads", L"Fatal Fissures", L"Custom…"
			});
			this->PresetDropdown->Location = System::Drawing::Point(12, 33);
			this->PresetDropdown->Name = L"PresetDropdown";
			this->PresetDropdown->Size = System::Drawing::Size(176, 29);
			this->PresetDropdown->TabIndex = 1;
			this->PresetDropdown->Text = L"Custom...";
			this->PresetDropdown->SelectedIndexChanged += gcnew System::EventHandler(this, &MainForm::PresetDropdownChanged);
			// 
			// PresetLabel
			// 
			this->PresetLabel->AutoSize = true;
			this->PresetLabel->Font = (gcnew System::Drawing::Font(L"Helvetica", 10.8F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->PresetLabel->Location = System::Drawing::Point(12, 9);
			this->PresetLabel->Name = L"PresetLabel";
			this->PresetLabel->Size = System::Drawing::Size(67, 21);
			this->PresetLabel->TabIndex = 0;
			this->PresetLabel->Text = L"Preset:";
			this->PresetLabel->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// MainForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(8, 16);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(982, 703);
			this->Controls->Add(this->ScreenDivider);
			this->DoubleBuffered = true;
			this->Margin = System::Windows::Forms::Padding(4);
			this->Name = L"MainForm";
			this->ShowIcon = false;
			this->Text = L"Accelerated 3D Fractal";
			this->WindowState = System::Windows::Forms::FormWindowState::Maximized;
			this->FormClosed += gcnew System::Windows::Forms::FormClosedEventHandler(this, &MainForm::OnClosed);
			this->Load += gcnew System::EventHandler(this, &MainForm::OnLoad);
			this->ScreenDivider->Panel1->ResumeLayout(false);
			this->ScreenDivider->Panel1->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->ScreenDivider))->EndInit();
			this->ScreenDivider->ResumeLayout(false);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->IterationsSlider))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->PhiSlider))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->ThetaSlider))->EndInit();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->ScaleSlider))->EndInit();
			this->ResumeLayout(false);

		}
#pragma endregion
		Void OnLoad(Object^ sender, EventArgs^ e);
		Void OnClosed(System::Object^ sender, System::Windows::Forms::FormClosedEventArgs^ e);
		Void PresetDropdownChanged(Object^ sender, EventArgs^ e);
		Void OnDraw(Object^ sender, PaintEventArgs^ e);
		Void OnMouseMove(Object^ sender, MouseEventArgs^ e);
		Void OnMouseDown(Object^ sender, MouseEventArgs^ e);
		Void OnMouseUp(Object^ sender, MouseEventArgs^ e);
		Void OnMouseWheel(Object^ sender, MouseEventArgs^ e);
		Void ScaleTextChanged(Object^ sender, EventArgs^ e);
		Void ScaleSliderScroll(Object^ sender, EventArgs^ e);
		Void ThetaTextChanged(Object^ sender, EventArgs^ e);
		Void ThetaSliderScroll(Object^ sender, EventArgs^ e);
		Void PhiTextChanged(Object^ sender, EventArgs^ e);
		Void PhiSliderScroll(Object^ sender, EventArgs^ e);
		Void OffsetXTextChanged(Object^ sender, EventArgs^ e);
		Void OffsetYTextChanged(Object^ sender, EventArgs^ e);
		Void OffsetZTextChanged(Object^ sender, EventArgs^ e);
		Void IterationsSliderScroll(Object^ sender, EventArgs^ e);
		Void SetScaleControls();
		Void SetThetaControls();
		Void SetPhiControls();
		Void SetOffsetControls();
		cli::array<unsigned char>^ pixel_values;
		Bitmap^ b;
	};
}

