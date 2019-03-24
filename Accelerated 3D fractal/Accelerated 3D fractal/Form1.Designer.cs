namespace Accelerated_3D_fractal
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.ScreenDivider = new System.Windows.Forms.SplitContainer();
            this.OffsetZLabel = new System.Windows.Forms.Label();
            this.OffsetYLabel = new System.Windows.Forms.Label();
            this.OffsetXLabel = new System.Windows.Forms.Label();
            this.OffsetZText = new System.Windows.Forms.TextBox();
            this.OffsetYText = new System.Windows.Forms.TextBox();
            this.MaxIterationsLabel = new System.Windows.Forms.Label();
            this.MinIterationsLabel = new System.Windows.Forms.Label();
            this.IterationsSlider = new System.Windows.Forms.TrackBar();
            this.IterationsLabel = new System.Windows.Forms.Label();
            this.OffsetXText = new System.Windows.Forms.TextBox();
            this.OffsetLabel = new System.Windows.Forms.Label();
            this.ScaleMaxLabel = new System.Windows.Forms.Label();
            this.ScaleMinLabel = new System.Windows.Forms.Label();
            this.ScaleText = new System.Windows.Forms.TextBox();
            this.Angle1Text = new System.Windows.Forms.TextBox();
            this.Angle2Text = new System.Windows.Forms.TextBox();
            this.MaxAngle2Label = new System.Windows.Forms.Label();
            this.MinAngle2Label = new System.Windows.Forms.Label();
            this.Angle2Slider = new System.Windows.Forms.TrackBar();
            this.Angle2Label = new System.Windows.Forms.Label();
            this.MaxAngle1Label = new System.Windows.Forms.Label();
            this.MinAngle1Label = new System.Windows.Forms.Label();
            this.Angle1Slider = new System.Windows.Forms.TrackBar();
            this.Angle1Label = new System.Windows.Forms.Label();
            this.ScaleSlider = new System.Windows.Forms.TrackBar();
            this.ScaleLabel = new System.Windows.Forms.Label();
            this.LevelDropdown = new System.Windows.Forms.ComboBox();
            this.PresetLabel = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.ScreenDivider)).BeginInit();
            this.ScreenDivider.Panel1.SuspendLayout();
            this.ScreenDivider.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.IterationsSlider)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.Angle2Slider)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.Angle1Slider)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.ScaleSlider)).BeginInit();
            this.SuspendLayout();
            // 
            // ScreenDivider
            // 
            this.ScreenDivider.Dock = System.Windows.Forms.DockStyle.Fill;
            this.ScreenDivider.Location = new System.Drawing.Point(0, 0);
            this.ScreenDivider.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.ScreenDivider.Name = "ScreenDivider";
            // 
            // ScreenDivider.Panel1
            // 
            this.ScreenDivider.Panel1.Controls.Add(this.OffsetZLabel);
            this.ScreenDivider.Panel1.Controls.Add(this.OffsetYLabel);
            this.ScreenDivider.Panel1.Controls.Add(this.OffsetXLabel);
            this.ScreenDivider.Panel1.Controls.Add(this.OffsetZText);
            this.ScreenDivider.Panel1.Controls.Add(this.OffsetYText);
            this.ScreenDivider.Panel1.Controls.Add(this.MaxIterationsLabel);
            this.ScreenDivider.Panel1.Controls.Add(this.MinIterationsLabel);
            this.ScreenDivider.Panel1.Controls.Add(this.IterationsSlider);
            this.ScreenDivider.Panel1.Controls.Add(this.IterationsLabel);
            this.ScreenDivider.Panel1.Controls.Add(this.OffsetXText);
            this.ScreenDivider.Panel1.Controls.Add(this.OffsetLabel);
            this.ScreenDivider.Panel1.Controls.Add(this.ScaleMaxLabel);
            this.ScreenDivider.Panel1.Controls.Add(this.ScaleMinLabel);
            this.ScreenDivider.Panel1.Controls.Add(this.ScaleText);
            this.ScreenDivider.Panel1.Controls.Add(this.Angle1Text);
            this.ScreenDivider.Panel1.Controls.Add(this.Angle2Text);
            this.ScreenDivider.Panel1.Controls.Add(this.MaxAngle2Label);
            this.ScreenDivider.Panel1.Controls.Add(this.MinAngle2Label);
            this.ScreenDivider.Panel1.Controls.Add(this.Angle2Slider);
            this.ScreenDivider.Panel1.Controls.Add(this.Angle2Label);
            this.ScreenDivider.Panel1.Controls.Add(this.MaxAngle1Label);
            this.ScreenDivider.Panel1.Controls.Add(this.MinAngle1Label);
            this.ScreenDivider.Panel1.Controls.Add(this.Angle1Slider);
            this.ScreenDivider.Panel1.Controls.Add(this.Angle1Label);
            this.ScreenDivider.Panel1.Controls.Add(this.ScaleSlider);
            this.ScreenDivider.Panel1.Controls.Add(this.ScaleLabel);
            this.ScreenDivider.Panel1.Controls.Add(this.LevelDropdown);
            this.ScreenDivider.Panel1.Controls.Add(this.PresetLabel);
            // 
            // ScreenDivider.Panel2
            // 
            this.ScreenDivider.Panel2.Paint += new System.Windows.Forms.PaintEventHandler(this.OnDraw);
            this.ScreenDivider.Panel2.MouseDown += new System.Windows.Forms.MouseEventHandler(this.OnMouseDown);
            this.ScreenDivider.Panel2.MouseMove += new System.Windows.Forms.MouseEventHandler(this.OnMouseMove);
            this.ScreenDivider.Panel2.MouseUp += new System.Windows.Forms.MouseEventHandler(this.OnMouseUp);
            this.ScreenDivider.Panel2.MouseWheel += new System.Windows.Forms.MouseEventHandler(this.OnMouseWheel);
            this.ScreenDivider.Size = new System.Drawing.Size(1104, 879);
            this.ScreenDivider.SplitterDistance = 366;
            this.ScreenDivider.TabIndex = 0;
            // 
            // OffsetZLabel
            // 
            this.OffsetZLabel.AutoSize = true;
            this.OffsetZLabel.Location = new System.Drawing.Point(46, 498);
            this.OffsetZLabel.Name = "OffsetZLabel";
            this.OffsetZLabel.Size = new System.Drawing.Size(23, 20);
            this.OffsetZLabel.TabIndex = 22;
            this.OffsetZLabel.Text = "Z:";
            // 
            // OffsetYLabel
            // 
            this.OffsetYLabel.AutoSize = true;
            this.OffsetYLabel.Location = new System.Drawing.Point(46, 462);
            this.OffsetYLabel.Name = "OffsetYLabel";
            this.OffsetYLabel.Size = new System.Drawing.Size(24, 20);
            this.OffsetYLabel.TabIndex = 20;
            this.OffsetYLabel.Text = "Y:";
            // 
            // OffsetXLabel
            // 
            this.OffsetXLabel.AutoSize = true;
            this.OffsetXLabel.Location = new System.Drawing.Point(46, 428);
            this.OffsetXLabel.Name = "OffsetXLabel";
            this.OffsetXLabel.Size = new System.Drawing.Size(24, 20);
            this.OffsetXLabel.TabIndex = 18;
            this.OffsetXLabel.Text = "X:";
            // 
            // OffsetZText
            // 
            this.OffsetZText.Location = new System.Drawing.Point(76, 494);
            this.OffsetZText.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.OffsetZText.Name = "OffsetZText";
            this.OffsetZText.Size = new System.Drawing.Size(76, 26);
            this.OffsetZText.TabIndex = 23;
            this.OffsetZText.Text = "0.000";
            this.OffsetZText.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // OffsetYText
            // 
            this.OffsetYText.Location = new System.Drawing.Point(76, 459);
            this.OffsetYText.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.OffsetYText.Name = "OffsetYText";
            this.OffsetYText.Size = new System.Drawing.Size(76, 26);
            this.OffsetYText.TabIndex = 21;
            this.OffsetYText.Text = "0.000";
            this.OffsetYText.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            // 
            // MaxIterationsLabel
            // 
            this.MaxIterationsLabel.AutoSize = true;
            this.MaxIterationsLabel.Location = new System.Drawing.Point(312, 604);
            this.MaxIterationsLabel.Name = "MaxIterationsLabel";
            this.MaxIterationsLabel.Size = new System.Drawing.Size(27, 20);
            this.MaxIterationsLabel.TabIndex = 27;
            this.MaxIterationsLabel.Text = "20";
            // 
            // MinIterationsLabel
            // 
            this.MinIterationsLabel.AutoSize = true;
            this.MinIterationsLabel.Location = new System.Drawing.Point(19, 604);
            this.MinIterationsLabel.Name = "MinIterationsLabel";
            this.MinIterationsLabel.Size = new System.Drawing.Size(18, 20);
            this.MinIterationsLabel.TabIndex = 26;
            this.MinIterationsLabel.Text = "0";
            // 
            // IterationsSlider
            // 
            this.IterationsSlider.Location = new System.Drawing.Point(9, 561);
            this.IterationsSlider.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.IterationsSlider.Maximum = 20;
            this.IterationsSlider.Name = "IterationsSlider";
            this.IterationsSlider.Size = new System.Drawing.Size(330, 69);
            this.IterationsSlider.TabIndex = 25;
            this.IterationsSlider.Scroll += new System.EventHandler(this.IterationsSlider_Scroll);
            // 
            // IterationsLabel
            // 
            this.IterationsLabel.AutoSize = true;
            this.IterationsLabel.Location = new System.Drawing.Point(14, 536);
            this.IterationsLabel.Name = "IterationsLabel";
            this.IterationsLabel.Size = new System.Drawing.Size(80, 20);
            this.IterationsLabel.TabIndex = 24;
            this.IterationsLabel.Text = "Iterations:";
            // 
            // OffsetXText
            // 
            this.OffsetXText.Location = new System.Drawing.Point(76, 424);
            this.OffsetXText.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.OffsetXText.Name = "OffsetXText";
            this.OffsetXText.Size = new System.Drawing.Size(76, 26);
            this.OffsetXText.TabIndex = 19;
            this.OffsetXText.Text = "0.000";
            this.OffsetXText.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.OffsetXText.TextChanged += new System.EventHandler(this.OffsetXText_TextChanged);
            // 
            // OffsetLabel
            // 
            this.OffsetLabel.AutoSize = true;
            this.OffsetLabel.Location = new System.Drawing.Point(14, 398);
            this.OffsetLabel.Name = "OffsetLabel";
            this.OffsetLabel.Size = new System.Drawing.Size(57, 20);
            this.OffsetLabel.TabIndex = 17;
            this.OffsetLabel.Text = "Offset:";
            // 
            // ScaleMaxLabel
            // 
            this.ScaleMaxLabel.AutoSize = true;
            this.ScaleMaxLabel.Location = new System.Drawing.Point(290, 142);
            this.ScaleMaxLabel.Name = "ScaleMaxLabel";
            this.ScaleMaxLabel.Size = new System.Drawing.Size(49, 20);
            this.ScaleMaxLabel.TabIndex = 6;
            this.ScaleMaxLabel.Text = "2.000";
            // 
            // ScaleMinLabel
            // 
            this.ScaleMinLabel.AutoSize = true;
            this.ScaleMinLabel.Location = new System.Drawing.Point(10, 142);
            this.ScaleMinLabel.Name = "ScaleMinLabel";
            this.ScaleMinLabel.Size = new System.Drawing.Size(49, 20);
            this.ScaleMinLabel.TabIndex = 5;
            this.ScaleMinLabel.Text = "0.000";
            // 
            // ScaleText
            // 
            this.ScaleText.Location = new System.Drawing.Point(74, 71);
            this.ScaleText.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.ScaleText.Name = "ScaleText";
            this.ScaleText.Size = new System.Drawing.Size(75, 26);
            this.ScaleText.TabIndex = 3;
            this.ScaleText.Text = "1.000";
            this.ScaleText.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.ScaleText.TextChanged += new System.EventHandler(this.ScaleText_TextChanged);
            // 
            // Angle1Text
            // 
            this.Angle1Text.Location = new System.Drawing.Point(88, 175);
            this.Angle1Text.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.Angle1Text.Name = "Angle1Text";
            this.Angle1Text.Size = new System.Drawing.Size(75, 26);
            this.Angle1Text.TabIndex = 8;
            this.Angle1Text.Text = "0.00";
            this.Angle1Text.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.Angle1Text.TextChanged += new System.EventHandler(this.Angle1Text_TextChanged);
            // 
            // Angle2Text
            // 
            this.Angle2Text.Location = new System.Drawing.Point(84, 275);
            this.Angle2Text.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.Angle2Text.Name = "Angle2Text";
            this.Angle2Text.Size = new System.Drawing.Size(75, 26);
            this.Angle2Text.TabIndex = 13;
            this.Angle2Text.Text = "0.00";
            this.Angle2Text.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.Angle2Text.VisibleChanged += new System.EventHandler(this.Angle2Text_TextChanged);
            // 
            // MaxAngle2Label
            // 
            this.MaxAngle2Label.AutoSize = true;
            this.MaxAngle2Label.Location = new System.Drawing.Point(322, 339);
            this.MaxAngle2Label.Name = "MaxAngle2Label";
            this.MaxAngle2Label.Size = new System.Drawing.Size(18, 20);
            this.MaxAngle2Label.TabIndex = 16;
            this.MaxAngle2Label.Text = "π";
            // 
            // MinAngle2Label
            // 
            this.MinAngle2Label.AutoSize = true;
            this.MinAngle2Label.Location = new System.Drawing.Point(15, 339);
            this.MinAngle2Label.Name = "MinAngle2Label";
            this.MinAngle2Label.Size = new System.Drawing.Size(23, 20);
            this.MinAngle2Label.TabIndex = 15;
            this.MinAngle2Label.Text = "-π";
            // 
            // Angle2Slider
            // 
            this.Angle2Slider.Location = new System.Drawing.Point(10, 305);
            this.Angle2Slider.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.Angle2Slider.Maximum = 314;
            this.Angle2Slider.Minimum = -314;
            this.Angle2Slider.Name = "Angle2Slider";
            this.Angle2Slider.Size = new System.Drawing.Size(330, 69);
            this.Angle2Slider.TabIndex = 14;
            this.Angle2Slider.TickStyle = System.Windows.Forms.TickStyle.None;
            this.Angle2Slider.Scroll += new System.EventHandler(this.Angle2Slider_Scroll);
            // 
            // Angle2Label
            // 
            this.Angle2Label.AutoSize = true;
            this.Angle2Label.Location = new System.Drawing.Point(10, 279);
            this.Angle2Label.Name = "Angle2Label";
            this.Angle2Label.Size = new System.Drawing.Size(67, 20);
            this.Angle2Label.TabIndex = 12;
            this.Angle2Label.Text = "Angle 2:";
            // 
            // MaxAngle1Label
            // 
            this.MaxAngle1Label.AutoSize = true;
            this.MaxAngle1Label.Location = new System.Drawing.Point(322, 238);
            this.MaxAngle1Label.Name = "MaxAngle1Label";
            this.MaxAngle1Label.Size = new System.Drawing.Size(18, 20);
            this.MaxAngle1Label.TabIndex = 11;
            this.MaxAngle1Label.Text = "π";
            // 
            // MinAngle1Label
            // 
            this.MinAngle1Label.AutoSize = true;
            this.MinAngle1Label.Location = new System.Drawing.Point(14, 238);
            this.MinAngle1Label.Name = "MinAngle1Label";
            this.MinAngle1Label.Size = new System.Drawing.Size(23, 20);
            this.MinAngle1Label.TabIndex = 10;
            this.MinAngle1Label.Text = "-π";
            // 
            // Angle1Slider
            // 
            this.Angle1Slider.Location = new System.Drawing.Point(10, 205);
            this.Angle1Slider.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.Angle1Slider.Maximum = 314;
            this.Angle1Slider.Minimum = -314;
            this.Angle1Slider.Name = "Angle1Slider";
            this.Angle1Slider.Size = new System.Drawing.Size(330, 69);
            this.Angle1Slider.TabIndex = 9;
            this.Angle1Slider.TickStyle = System.Windows.Forms.TickStyle.None;
            this.Angle1Slider.Scroll += new System.EventHandler(this.Angle1Slider_Scroll);
            // 
            // Angle1Label
            // 
            this.Angle1Label.AutoSize = true;
            this.Angle1Label.Location = new System.Drawing.Point(14, 179);
            this.Angle1Label.Name = "Angle1Label";
            this.Angle1Label.Size = new System.Drawing.Size(67, 20);
            this.Angle1Label.TabIndex = 7;
            this.Angle1Label.Text = "Angle 1:";
            // 
            // ScaleSlider
            // 
            this.ScaleSlider.Location = new System.Drawing.Point(10, 100);
            this.ScaleSlider.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.ScaleSlider.Maximum = 2000;
            this.ScaleSlider.Name = "ScaleSlider";
            this.ScaleSlider.Size = new System.Drawing.Size(330, 69);
            this.ScaleSlider.TabIndex = 4;
            this.ScaleSlider.TickStyle = System.Windows.Forms.TickStyle.None;
            this.ScaleSlider.Value = 1000;
            this.ScaleSlider.Scroll += new System.EventHandler(this.ScaleSlider_Scroll);
            // 
            // ScaleLabel
            // 
            this.ScaleLabel.AutoSize = true;
            this.ScaleLabel.Location = new System.Drawing.Point(15, 75);
            this.ScaleLabel.Name = "ScaleLabel";
            this.ScaleLabel.Size = new System.Drawing.Size(53, 20);
            this.ScaleLabel.TabIndex = 2;
            this.ScaleLabel.Text = "Scale:";
            // 
            // LevelDropdown
            // 
            this.LevelDropdown.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.LevelDropdown.FormattingEnabled = true;
            this.LevelDropdown.Items.AddRange(new object[] {
            "Jump The Crater",
            "Too Many Trees",
            "Hole In One",
            "Around The World",
            "The Hills Are Alive",
            "Beware Of Bumps",
            "Mountain Climbing",
            "The Catwalk",
            "Mind The Gap",
            "Don\'t Get Crushed",
            "The Sponge",
            "Ride The Gecko",
            "Build Up Speed",
            "Around The Citadel",
            "Planet Crusher",
            "Top Of The Citadel",
            "Building Bridges",
            "Pylon Palace",
            "The Crown Jewels",
            "Expressways",
            "Bunny Hops",
            "Asteroid Field",
            "Lily Pads",
            "Fatal Fissures",
            "Custom..."});
            this.LevelDropdown.Location = new System.Drawing.Point(17, 36);
            this.LevelDropdown.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.LevelDropdown.Name = "LevelDropdown";
            this.LevelDropdown.Size = new System.Drawing.Size(216, 28);
            this.LevelDropdown.TabIndex = 1;
            this.LevelDropdown.SelectedIndexChanged += new System.EventHandler(this.IndexChanged);
            // 
            // PresetLabel
            // 
            this.PresetLabel.AutoSize = true;
            this.PresetLabel.Location = new System.Drawing.Point(14, 11);
            this.PresetLabel.Name = "PresetLabel";
            this.PresetLabel.Size = new System.Drawing.Size(59, 20);
            this.PresetLabel.TabIndex = 0;
            this.PresetLabel.Text = "Preset:";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 20F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1104, 879);
            this.Controls.Add(this.ScreenDivider);
            this.Margin = new System.Windows.Forms.Padding(4, 5, 4, 5);
            this.Name = "Form1";
            this.ShowIcon = false;
            this.SizeGripStyle = System.Windows.Forms.SizeGripStyle.Hide;
            this.Text = "Accelerated 3D fractal";
            this.WindowState = System.Windows.Forms.FormWindowState.Maximized;
            this.Load += new System.EventHandler(this.OnLoad);
            this.ScreenDivider.Panel1.ResumeLayout(false);
            this.ScreenDivider.Panel1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.ScreenDivider)).EndInit();
            this.ScreenDivider.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.IterationsSlider)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.Angle2Slider)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.Angle1Slider)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.ScaleSlider)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.SplitContainer ScreenDivider;
        private System.Windows.Forms.ComboBox LevelDropdown;
        private System.Windows.Forms.Label PresetLabel;
        private System.Windows.Forms.TextBox Angle1Text;
        private System.Windows.Forms.TextBox Angle2Text;
        private System.Windows.Forms.Label MaxAngle2Label;
        private System.Windows.Forms.Label MinAngle2Label;
        private System.Windows.Forms.TrackBar Angle2Slider;
        private System.Windows.Forms.Label Angle2Label;
        private System.Windows.Forms.Label MaxAngle1Label;
        private System.Windows.Forms.Label MinAngle1Label;
        private System.Windows.Forms.TrackBar Angle1Slider;
        private System.Windows.Forms.Label Angle1Label;
        private System.Windows.Forms.TrackBar ScaleSlider;
        private System.Windows.Forms.Label ScaleLabel;
        private System.Windows.Forms.TextBox ScaleText;
        private System.Windows.Forms.Label ScaleMaxLabel;
        private System.Windows.Forms.Label ScaleMinLabel;
        private System.Windows.Forms.Label MaxIterationsLabel;
        private System.Windows.Forms.Label MinIterationsLabel;
        private System.Windows.Forms.TrackBar IterationsSlider;
        private System.Windows.Forms.Label IterationsLabel;
        private System.Windows.Forms.TextBox OffsetXText;
        private System.Windows.Forms.Label OffsetLabel;
        private System.Windows.Forms.TextBox OffsetZText;
        private System.Windows.Forms.TextBox OffsetYText;
        private System.Windows.Forms.Label OffsetZLabel;
        private System.Windows.Forms.Label OffsetYLabel;
        private System.Windows.Forms.Label OffsetXLabel;
    }
}

