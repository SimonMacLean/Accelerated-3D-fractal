using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Alea;
using Alea.CSharp;
using Alea.FSharp;
using Alea.Parallel;

namespace Accelerated_3D_fractal
{
    public partial class Form1 : Form
    {
        private Bitmap b;
        private byte[] pixels;
        private DeviceMemory<byte> pixDevMem;
        private deviceptr<byte> pixDevPtr;
        private float3[] directions;
        private DeviceMemory<float3> dirDevMem;
        private deviceptr<float3> dirDevPtr;
        private float3 camera;
        private float cameraBaseDist;
        private float3 center;
        private float3 lightLocation;
        private float3 x;
        private float3 y;
        private float3 z;
        private Timer t;
        private float elapsed = 0;
        private float baseMovementSize = 1f;
        private float movementSize;
        private string levelName = "Custom";
        private float3 seed = new float3(1, 0, 0);
        private float3 shift = new float3(0, 0, 0);
        private float2 cols = new float2(0.3f, 0.8f);
        private string[] names;
        private float3[] seeds;
        private float3[] offsets;
        private int levelnum = -1;
        private int2 mouseLocation;
        private float mouseSensitivity = 0.01f;
        private bool isMouseDown = false;
        private float focalLength = -2;
        private float minDist = 0.0001f;
        private float maxDist = 100;
        private int granularity = 1;
        private int iterations = 0;
        private float side = 6;
        private int width;
        private int height;
        private int bytes = 3;
        private float ambientOccStrength = 0.05f;
        private float shadowStrength = 8;
        private readonly dim3 BlockSize = new dim3(32, 32);
        private dim3 GridSize;
        private LaunchParam launchParam;
        private Gpu gpu;
        public Form1()
        {
            InitializeComponent();
            gpu = Gpu.Default;
            DoubleBuffered = true;
        }
        public void Init()
        {
            Font = new Font(FontFamily.GenericMonospace, 10);
            width = ClientRectangle.Width / granularity / 2 * 2;
            height = ClientRectangle.Height / granularity / 2 * 2;
            directions = new float3[width * height];
            dirDevMem = gpu.AllocateDevice(directions);
            dirDevPtr = dirDevMem.Ptr;
            pixels = new byte[width * height * bytes];
            pixDevMem = gpu.AllocateDevice(pixels);
            pixDevPtr = pixDevMem.Ptr;
            b = new Bitmap(width, height, width,
                     PixelFormat.Format24bppRgb,
                     Marshal.UnsafeAddrOfPinnedArrayElement(pixels, 0));
            center = new float3(0, 0, 0);
            camera = new float3(7, 0, 0);
            cameraBaseDist = 1.5f;
            lightLocation = new float3(20, 20, 20);
            GridSize = new dim3(width / BlockSize.x, height / BlockSize.y);
            launchParam = new LaunchParam(GridSize, BlockSize);
            movementSize = baseMovementSize;
            GetDirections();
            x = new float3(0, 0, 1);
            y = new float3(0, 1, 0);
            z = new float3(-1, 0, 0);
            names = new[]{
                "Jump The Crater",
                "Too Many Trees",
                "Hole In One",
                "Around The World",
                "The Hills Are Alive",
                "Beware Of Bumps",
                "Mountain Climbing",
                "The Catwalk",
                "Mind The Gap",
                "Don't Get Crushed",
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
                "Fatal Fissures"
            };
            seeds = new[]{
                new float3(1.8f, -0.12f, 0.5f),
                new float3(1.9073f, 2.72f, -1.16f),
                new float3(2.02f, -1.57f, 1.62f),
                new float3(1.65f, 0.37f, 5.26f),
                new float3(1.77f, -0.22f, 5.62f),
                new float3(1.66f, 1.52f, 0.19f),
                new float3(1.58f, -1.45f, 3.95f),
                new float3(1.87f, -3.12f, 0.02f),
                new float3(1.81f, -4.84f, -2.99f),
                new float3(1.93f, 1.34637f, 1.58f),
                new float3(1.88f, 1.52f, 4.91f),
                new float3(1.6f, 3.77f, 3.93f),
                new float3(2.08f, -4.79f, 3.16f),
                new float3(2.0773f, -9.66f, -1.34f),
                new float3(1.78f, -0.1f, 3.28f),
                new float3(2.0773f, -9.66f, -1.34f),
                new float3(1.8093f, -3.165f, -3.2094777f),
                new float3(1.95f, 1.570796f, 0),
                new float3(1.91f, 0.06f, -0.76f),
                new float3(1.8986f, -0.4166f, 0.00683f),
                new float3(2.03413f, 1.688f, -1.57798f),
                new float3(1.6516888f, 0.026083898f, -0.7996324f),
                new float3(1.77746f, 4.62318f, 0.0707307f),
                new float3(2.13f, -1.77f, -1.62f)
            };
            offsets = new[]{
                new float3(-2.12f, -2.75f, 0.49f),
                new float3(-2.958f, -3.193f, 2.695f),
                new float3(-3.31f, 6.19f, 1.53f),
                new float3(-1.41f, -0.22f, -0.77f),
                new float3(-2.08f, -1.42f, -1.93f),
                new float3(-3.83f, -1.94f, -1.09f),
                new float3(-1.55f, -0.13f, -2.52f),
                new float3(-3.57f, 0.129f, 2.95f),
                new float3(-2.905f, 0.765f, -4.165f),
                new float3(-2.31f, 1.123f, 1.56f),
                new float3(-4.54f, -1.26f, 0.1f),
                new float3(-2, -0.41f, -1.43f),
                new float3(-7.43f, 5.96f, -6.23f),
                new float3(-1.238f, -1.533f, 1.085f),
                new float3(-1.47f, 1.7f, -0.4f),
                new float3(-1.238f, -1.533f, 1.085f),
                new float3(-1.0939f, -0.43495f, -3.1113f),
                new float3(-6.75f, -3, 0),
                new float3(-3.44f, -0.69f, -1.14f),
                new float3(-2.5130f, -5.4067f, -2.51f),
                new float3(-4.803822f, -4.1f, -1.39063f),
                new float3(-3.85863f, -5.13741f, -0.918303f),
                new float3(-4.6867f, -0.84376f, 1.98158f),
                new float3(-4.99f, -3.05f, -4.48f)
            };
        }
        public static float L(float3 p)
        {
            return DeviceFunction.Sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        }
        public static float O(float3 a, float3 b)
        {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }
        public static float3 D(float3 p, float d)
        {
            p.x /= d;
            p.y /= d;
            p.z /= d;
            return p;
        }
        public static float3 D(float3 p, float3 d)
        {
            p.x /= d.x;
            p.y /= d.y;
            p.z /= d.z;
            return p;
        }
        public static float3 M(float3 p, float d)
        {
            p.x *= d;
            p.y *= d;
            p.z *= d;
            return p;
        }
        public static float3 M(float3 p, float3 d)
        {
            p.x *= d.x;
            p.y *= d.y;
            p.z *= d.z;
            return p;
        }
        public static float3 A(float3 a, float3 b)
        {
            a.x += b.x;
            a.y += b.y;
            a.z += b.z;
            return a;
        }
        public static float3 A(float3 a, float b)
        {
            a.x += b;
            a.y += b;
            a.z += b;
            return a;
        }
        public static float3 S(float3 a, float3 b)
        {
            a.x -= b.x;
            a.y -= b.y;
            a.z -= b.z;
            return a;
        }
        public static float3 S(float3 a, float b)
        {
            a.x -= b;
            a.y -= b;
            a.z -= b;
            return a;
        }
        public static float3 R(float3 a, float3 b)
        {
            a.x %= b.x;
            a.y %= b.y;
            a.z %= b.z;
            return a;
        }
        public static int Index(int x, int y, int width)
        {
            return y * width + x;
        }
        public void GetDirection(deviceptr<float3> directions, float focalLength, float width, float height)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;
            int h = Index(i, (int)height - 1 - j, (int)width);
            float3 p = new float3(focalLength, (j - height / 2) / height, (i - width / 2) / height);
            p = D(p, L(p));
            directions[h] = p;
        }
        public void RotateDirection(deviceptr<float3> directions, float3 a, float t, float c, float s, int width, int height)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;
            int h = Index(i, height - 1 - j, width);
            directions[h] = RotateVec(directions[h], a, t, c, s);
            directions[h] = D(directions[h], L(directions[h]));
        }
        public void GetColor(deviceptr<Color> colors)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            colors[i] = Color.FromArgb(i, i, i);
        }
        public static void MarchRay(deviceptr<float3> directions, deviceptr<byte> pixelValues, float3 camera,
            float3 light, float2 cols, float minDist, float maxDist, int maxstep, int bytes, int width, int iterations,
            float side, float3 seed, float3 shift, float shadowStrength, float ambientOccStrength)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;
            int h = Index(i, j, width);
            float3 p = camera;
            int stepnum = 0;
            float dist = minDist + 1;
            float totalDist = 0;
            while (totalDist < maxDist && dist > minDist)
            {
                dist = ScaledDE(p, iterations, side, seed, shift);
                p = A(p, M(directions[h], dist));
                totalDist += dist;
                stepnum++;
                if (stepnum == maxstep)
                    break;
            }
            float brightness = 0;
            if (DeviceFunction.Abs(dist) <= minDist)
            {
                float3 off = S(light, p);
                float lightVectorLength = L(off);
                off = D(off, lightVectorLength);
                float shadow = 1;
                float diffuseCalculated = 0;
                float normalAngle = O(off, Normal(p, iterations, side, seed, shift, minDist));
                if (normalAngle > 0)
                {
                    shadow = NewSoftShadow(p, off, shadowStrength, iterations, side, seed, shift, minDist, lightVectorLength, 0.01f);
                    diffuseCalculated = DeviceFunction.Max(cols.y * shadow * normalAngle, 0);
                }
                brightness += diffuseCalculated + cols.x / (1 + stepnum * ambientOccStrength);
                brightness = DeviceFunction.Min(DeviceFunction.Max(brightness, 0), 1);
                float col = Orbit(p, iterations, side, seed, shift);
                pixelValues[h * 3] = (byte)(Blue(col) * brightness * byte.MaxValue);
                pixelValues[h * 3 + 1] = (byte)(Green(col) * brightness * byte.MaxValue);
                pixelValues[h * 3 + 2] = (byte)(Red(col) * brightness * byte.MaxValue);
            }
            else
            {
                pixelValues[h * 3] = 0;
                pixelValues[h * 3 + 1] = 0;
                pixelValues[h * 3 + 2] = 0;
            }
        }
        public static float TrapezoidWave(float loc)
        {
            return DeviceFunction.Min(DeviceFunction.Max(DeviceFunction.Abs(loc - 3), 0) - 1, 1);
        }
        public static float Red(float loc)
        {
            return TrapezoidWave(loc % 6);
        }
        public static float Green(float loc)
        {
            return TrapezoidWave((loc + 4) % 6);
        }
        public static float Blue(float loc)
        {
            return TrapezoidWave((loc + 2) % 6);
        }
        public void GetDirections()
        {
            gpu.Launch(GetDirection, launchParam, dirDevPtr, focalLength, (float)width, (float)height);
            Gpu.Copy(dirDevMem, directions);
        }
        [GpuManaged]
        public void MarchRays()
        {
            gpu.Launch(MarchRay, launchParam, dirDevPtr, pixDevPtr, camera, lightLocation, cols, minDist, maxDist, 1000,
                bytes, width, iterations, side, seed, shift, shadowStrength, ambientOccStrength);
            Gpu.Copy(pixDevMem, pixels);
            b = new Bitmap(width, height, width * 3,
                     PixelFormat.Format24bppRgb,
                     Marshal.UnsafeAddrOfPinnedArrayElement(pixels, 0));
            //for (int i = 0; i < Width * Height; i++)
            //{
            //   int greyscale = (int)(pixels[i] * 255);
            //    b.SetPixel(i % Width, Height - 1 - i / Width, Color.FromArgb(greyscale, greyscale, greyscale));
            //}
        }
        public static float OldSoftShadow(float3 p, float3 d, float shadowStrength, int iterations, float side, float3 seed, float3 shift, float minDist, float maxDist, float minAngle)
        {
            float k = 1;
            float dist = minDist;
            float angle = 1;
            float totalDist = minDist / 100;
            float3 marchedPoint = p;
            while (totalDist < maxDist)
            {
                dist = ScaledDE(marchedPoint, iterations, side, seed, shift);
                if (dist == 0)
                    dist = minDist;
                marchedPoint = A(marchedPoint, M(d, dist));
                totalDist += dist;
                angle = shadowStrength * dist / totalDist;
                k = DeviceFunction.Min(k, angle);
                if (dist < 0)
                    return 0;
                if (k < minAngle)
                {
                    return 0;
                }
            }
            return k;
        }
        public static float NewSoftShadow(float3 p, float3 d, float shadowStrength, int iterations, float side, float3 seed, float3 shift, float minDist, float maxDist, float minAngle)
        {
            float darkness = 1;
            float prevDist = float.MaxValue;
            float dist = minDist / 100;
            float angle = 1;
            float totalDist = minDist;
            float oldNewIntDist = 0;
            float legLength = 0;
            while (totalDist < maxDist)
            {
                dist = ScaledDE(A(p, M(d, totalDist)), iterations, side, seed, shift);
                oldNewIntDist = dist * dist / (2 * prevDist);
                legLength = DeviceFunction.Sqrt(dist * dist - oldNewIntDist * oldNewIntDist);
                angle = shadowStrength * legLength / DeviceFunction.Max(0, totalDist - oldNewIntDist);
                darkness = DeviceFunction.Min(darkness, angle);
                prevDist = dist;
                totalDist += dist;
                if (dist < 0)
                    return 0;
                if (darkness < minAngle)
                {
                    return 0;
                }
            }
            return darkness;
        }
        public static float3 Normal(float3 p, int iterations, float side, float3 seed, float3 shift, float epsilon)
        {
            float3 scaled = new float3(
                ScaledDE(new float3(p.x + epsilon, p.y, p.z), iterations, side, seed, shift) -
                ScaledDE(new float3(p.x - epsilon, p.y, p.z), iterations, side, seed, shift),
                ScaledDE(new float3(p.x, p.y + epsilon, p.z), iterations, side, seed, shift) -
                ScaledDE(new float3(p.x, p.y - epsilon, p.z), iterations, side, seed, shift),
                ScaledDE(new float3(p.x, p.y, p.z + epsilon), iterations, side, seed, shift) -
                ScaledDE(new float3(p.x, p.y, p.z - epsilon), iterations, side, seed, shift));
            return D(scaled, L(scaled));
        }
        public static float DE(float3 p, float side)
        {
            return CubeDE(p, new float3(0, 0, 0), side);
        }
        public static float ScaledDE(float3 p, int iterations, float side, float3 seed, float3 shift)
        {
            return WarpDist(DE(WarpSpace(p, iterations, seed, shift), side), iterations, seed.x);
        }
        public static float SphereDE(float3 p, float3 c, float di)
        {
            return L(S(p, c)) - di / 2;
        }
        public static float CubeDE(float3 p, float3 c, float di)
        {
            float3 o = S(p, c);
            float d = DeviceFunction.Max(DeviceFunction.Abs(o.x), DeviceFunction.Max(DeviceFunction.Abs(o.y), DeviceFunction.Abs(o.z)));
            return d - di / 2;
        }
        public static float3 Transform(float3 p, int iterations, float3 seed, float3 shift)
        {
            p = ScaleSpace(p, seed.x);
            p = AbsSpace(p);
            p = RotateZ(p, seed.y);
            p = FoldMenger(p);
            p = RotateX(p, seed.z);
            p = A(p, shift);
            return p;
        }
        public static float Orbit(float3 p, int iterations, float side, float3 seed, float3 shift)
        {
            float direction = ScaledDE(p, 1, side, seed, shift);
            for (int i = 0; i < iterations; i++)
            {
                p = Transform(p, iterations, seed, shift);
                if (WarpDist(DE(p, side), i, seed.x) * direction >= 0)
                {
                    return WarpDist(DE(p, side), i - 1, seed.x);
                }
            }
            return WarpDist(DE(p, side), iterations - 1, seed.x);
        }
        public static float3 WarpSpace(float3 p, int iterations, float3 seed, float3 shift)
        {
            for (int i = 0; i < iterations; i++)
            {
                p = Transform(p, iterations, seed, shift);
            }
            return p;
        }
        public static float WarpDist(float d, int iterations, float scale)
        {
            for (int i = 0; i < iterations; i++)
            {
                d = ScaleDist(d, scale);
            }
            return d;
        }
        public static float ScaleDist(float d, float s)
        {
            return d / s;
        }
        public static float3 ScaleSpace(float3 p, float s)
        {
            return M(p, s);
        }
        public static float3 ScaleSpace(float3 p, float3 s)
        {
            p.x *= s.x;
            p.y *= s.y;
            p.z *= s.z;
            return p;
        }
        public static float3 TranslateSpace(float3 p, float3 offset)
        {
            return S(p, offset);
        }
        public static float3 ModSpace(float3 p, float3 mod)
        {
            return S(R(A(R(A(p, D(mod, 2)), mod), mod), mod), D(mod, 2));
        }
        public static float3 FoldSpace(float3 p, float3 n)
        {
            if (O(p, n) >= 0)
                return p;
            else
                return S(p, D(M(M(n, 2), O(p, n)), O(n, n)));
        }
        public static float3 FoldMenger(float3 z)
        {
            float a = DeviceFunction.Min(z.x - z.y, 0);
            z.x -= a;
            z.y += a;
            a = DeviceFunction.Min(z.x - z.z, 0);
            z.x -= a;
            z.z += a;
            a = DeviceFunction.Min(z.y - z.z, 0);
            z.y -= a;
            z.z += a;
            return z;
        }
        public static float3 FoldBox(float3 z, float r)
        {
            return S(M(MaxSpace(MinSpace(z, r), -r), 2), z);
        }
        public static float3 MaxSpace(float3 a, float3 b)
        {
            a.x = DeviceFunction.Max(a.x, b.x);
            a.y = DeviceFunction.Max(a.y, b.y);
            a.z = DeviceFunction.Max(a.z, b.z);
            return a;
        }
        public static float3 MaxSpace(float3 a, float b)
        {
            a.x = DeviceFunction.Max(a.x, b);
            a.y = DeviceFunction.Max(a.y, b);
            a.z = DeviceFunction.Max(a.z, b);
            return a;
        }
        public static float3 MinSpace(float3 a, float3 b)
        {
            a.x = DeviceFunction.Min(a.x, b.x);
            a.y = DeviceFunction.Min(a.y, b.y);
            a.z = DeviceFunction.Min(a.z, b.z);
            return a;
        }
        public static float3 MinSpace(float3 a, float b)
        {
            a.x = DeviceFunction.Min(a.x, b);
            a.y = DeviceFunction.Min(a.y, b);
            a.z = DeviceFunction.Min(a.z, b);
            return a;
        }
        public static float3 AbsSpace(float3 p)
        {
            return new float3(DeviceFunction.Abs(p.x), DeviceFunction.Abs(p.y), DeviceFunction.Abs(p.z));
        }
        public static float3 AbsSpaceX(float3 p)
        {
            return new float3(DeviceFunction.Abs(p.x), p.y, p.z);
        }
        public static float3 AbsSpaceY(float3 p)
        {
            return new float3(p.x, DeviceFunction.Abs(p.y), p.z);
        }
        public static float3 AbsSpaceZ(float3 p)
        {
            return new float3(p.x, p.y, DeviceFunction.Abs(p.z));
        }
        public static float3 RotateX(float3 z, float t)
        {
            float3 p = z;
            float s = DeviceFunction.Sin(t);
            float c = DeviceFunction.Cos(t);
            p.y = c * z.y + s * z.z;
            p.z = c * z.z - s * z.y;
            return p;
        }
        public static float3 RotateY(float3 z, float t)
        {
            float3 p = z;
            float s = DeviceFunction.Sin(t);
            float c = DeviceFunction.Cos(t);
            p.x = c * z.x - s * z.z;
            p.z = c * z.z + s * z.x;
            return p;
        }
        public static float3 RotateZ(float3 z, float t)
        {
            float3 p = z;
            float s = DeviceFunction.Sin(t);
            float c = DeviceFunction.Cos(t);
            p.x = c * z.x + s * z.y;
            p.y = c * z.y - s * z.x;
            return p;
        }
        private void Form1_Load(object sender, EventArgs e)
        {
            Init();
            Invalidate();
        }
        private void Form1_Paint(object sender, PaintEventArgs e)
        {
            if (directions.Length != 0)
            {
                MarchRays();
                e.Graphics.DrawImage(b, 0, 0, width * granularity, height * granularity);
                e.Graphics.DrawString(
                string.Format(
@"Preset: {0} 
('D' to see next, 'A' to see previous, 'R' to reset)

Scale: {1} 
('>' to increase, '<' to decrease)

Angle1: {2} 
(')' to increase, '(' to decrease)

Angle2: {3} 
(']' to increase, '[' to decrease)

Offset: ({4}, {5}, {6}) 
(Arrow keys, Page Up, Page Down to change)

Iterations: {7} 
('+' to increase, '-' to decrease)

Camera Location: ({8}, {9}, {10})
(Scroll to move, click and drag to rotate)
'Esc' to exit", levelName, 1 / seed.x, seed.y, seed.z, shift.x, shift.y, shift.z, iterations, camera.x, camera.y, camera.z),
                     Font, Brushes.White, new Point(10, 10));
            }
        }
        private void Form1_KeyDown(object sender, KeyEventArgs e)
        {
            switch (e.KeyCode)
            {
                case Keys.D1:
                    seed.y += 0.05f;
                    Name = "Custom";
                    break;
                case Keys.D2:
                    seed.y -= 0.05f;
                    Name = "Custom";
                    break;
                case Keys.D3:
                    seed.z += 0.05f;
                    Name = "Custom";
                    break;
                case Keys.D4:
                    seed.z -= 0.05f;
                    Name = "Custom";
                    break;
                case Keys.Up:
                    shift.z += 0.05f;
                    Name = "Custom";
                    break;
                case Keys.Down:
                    shift.z -= 0.05f;
                    Name = "Custom";
                    break;
                case Keys.Left:
                    shift.x -= 0.05f;
                    Name = "Custom";
                    break;
                case Keys.Right:
                    shift.x += 0.05f;
                    Name = "Custom";
                    break;
                case Keys.PageUp:
                    shift.y += 0.05f;
                    Name = "Custom";
                    break;
                case Keys.PageDown:
                    shift.y -= 0.05f;
                    Name = "Custom";
                    break;
                case Keys.OemPeriod:
                    seed.x = 1 / (1 / seed.x + 0.1f / iterations);
                    Name = "Custom";
                    break;
                case Keys.Oemcomma:
                    seed.x = 1 / (1 / seed.x - 0.1f / iterations);
                    Name = "Custom";
                    break;
                case Keys.OemMinus:
                    iterations--;
                    break;
                case Keys.Oemplus:
                    iterations++;
                    break;
                case Keys.R:
                    seed.y = 0;
                    seed.z = 0;
                    shift = new float3(0, 0, 0);
                    seed.x = 1;
                    iterations = 1;
                    Name = "Custom";
                    break;
                case Keys.Space:
                    levelnum = (levelnum + 1) % names.Length;
                    levelName = names[levelnum];
                    seed = seeds[levelnum];
                    shift = offsets[levelnum];
                    break;
                case Keys.Escape:

                default:
                    return;
            }
            cameraBaseDist = L(camera);
            //minDist = 1f / Width * cameraBaseDist;
            if (minDist <= 0)
                minDist = 1;
            Invalidate();
        }
        private void Form1_MouseDown(object sender, MouseEventArgs e)
        {
            mouseLocation = new int2(e.X, e.Y);
            isMouseDown = true;
        }
        private void Form1_MouseUp(object sender, MouseEventArgs e)
        {
            isMouseDown = false;
        }
        [GpuManaged]
        private void Form1_MouseMove(object sender, MouseEventArgs e)
        {
            if (isMouseDown)
            {
                float2 offset = new float2((e.X - mouseLocation.x) * mouseSensitivity, (e.Y - mouseLocation.y) * mouseSensitivity);
                float c1 = DeviceFunction.Cos(offset.x);
                float s1 = DeviceFunction.Sin(offset.x);
                float t1 = 1 - c1;
                float c2 = DeviceFunction.Cos(offset.y);
                float s2 = DeviceFunction.Sin(offset.y);
                float t2 = 1 - c2;
                float camDist = L(camera);
                gpu.Launch(RotateDirection, launchParam, dirDevPtr, y, t1, c1, s1, width, height);
                x = RotateVec(x, y, t1, c1, s1);
                z = RotateVec(z, y, t1, c1, s1);
                camera = RotateVec(camera, y, t1, c1, s1);
                gpu.Launch(RotateDirection, launchParam, dirDevPtr, x, t2, c2, s2, width, height);
                y = RotateVec(y, x, t2, c2, s2);
                z = RotateVec(z, x, t2, c2, s2);
                camera = RotateVec(camera, x, t2, c2, s2);
                Gpu.Copy(dirDevMem, directions);
                x = D(x, L(x));
                y = D(y, L(y));
                z = D(z, L(z));
                Invalidate();
            }
            mouseLocation = new int2(e.X, e.Y);
        }
        private void Form1_MouseWheel(object sender, MouseEventArgs e)
        {
            if (mouseLocation.x >= 0 && mouseLocation.y >= 0 && mouseLocation.x < width * granularity && mouseLocation.y < height * granularity)
            {
                camera = A(camera, M(directions[Index(mouseLocation.x / granularity, mouseLocation.y / granularity, width)], movementSize * e.Delta / 100));
                Invalidate();
            }
        }
        public static float3 RotateVec(float3 p, float3 a, float t, float c, float s)
        {
            float d = O(a, p);
            return new float3(
                t * d * a.x + p.x * c + s * (a.y * p.z - a.z * p.y),
                t * d * a.y + p.y * c + s * (a.z * p.x - a.x * p.z),
                t * d * a.z + p.z * c + s * (a.x * p.y - a.y * p.x));
        }
    }
}