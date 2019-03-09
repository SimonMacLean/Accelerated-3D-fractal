using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using Alea;
using Alea.CSharp;
using Alea.FSharp;

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
        private float3 center;
        private float3 lightLocation;
        private float3 x;
        private float3 y;
        private float3 z;
        private float baseMovementSize = 1f;
        private float movementSize;
        private float3 seed = new float3(1, 0, 0);
        private float3 shift = new float3(0, 0, 0);
        private float2 cols = new float2(0.5f, 0.8f);
        private float3[] seeds;
        private float3[] offsets;
        private int2 mouseLocation;
        private float mouseSensitivity = 0.01f;
        private bool isMouseDown = false;
        private float focalLength = -2;
        private float minDist = 0.0001f;
        private float maxDist = 100;
        private int granularity = 1;
        private int iterations = 0;
        private float side = 1;
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
            width = ScreenDivider.Panel2.ClientSize.Width / granularity / 4 * 4;
            height = ScreenDivider.Panel2.ClientSize.Height / granularity / 4 * 4;
            directions = new float3[width * height];
            dirDevMem = gpu.AllocateDevice(directions);
            dirDevPtr = dirDevMem.Ptr;
            pixels = new byte[width * height * bytes];
            pixDevMem = gpu.AllocateDevice(pixels);
            pixDevPtr = pixDevMem.Ptr;
            b = new Bitmap(width, height, width * bytes,
                     PixelFormat.Format24bppRgb,
                     Marshal.UnsafeAddrOfPinnedArrayElement(pixels, 0));
            center = new float3(0, 0, 0);
            camera = new float3(3, 0, 0);
            lightLocation = new float3(20, 20, 20);
            GridSize = new dim3(width / BlockSize.x, height / BlockSize.y);
            launchParam = new LaunchParam(GridSize, BlockSize);
            movementSize = baseMovementSize;
            GetDirections();
            x = new float3(0, 0, 1);
            y = new float3(0, 1, 0);
            z = new float3(-1, 0, 0);
            seeds = new[]{
                new float3(1.8f, -0.12f, 0.5f),
                new float3(1.9073f, 2.72f, -1.16f),
                new float3(2.02f, -1.57f, 1.62f),
                new float3(1.65f, 0.37f, -1.023f),
                new float3(1.77f, -0.22f, -0.663f),
                new float3(1.66f, 1.52f, 0.19f),
                new float3(1.58f, -1.45f, -2.333f),
                new float3(1.87f, 3.141f, 0.02f),
                new float3(1.81f, 1.44f, -2.99f),
                new float3(1.93f, 1.34637f, 1.58f),
                new float3(1.88f, 1.52f, -1.373f),
                new float3(1.6f, -2.51f, -2.353f),
                new float3(2.08f, 1.493f, 3.141f),
                new float3(2.0773f, 2.906f, -1.34f),
                new float3(1.78f, -0.1f, -3.003f),
                new float3(2.0773f, 2.906f, -1.34f),
                new float3(1.8093f, 3.141f, 3.074f),
                new float3(1.95f, 1.570796f, 0),
                new float3(1.91f, 0.06f, -0.76f),
                new float3(1.8986f, -0.4166f, 0.00683f),
                new float3(2.03413f, 1.688f, -1.57798f),
                new float3(1.6516888f, 0.026083898f, -0.7996324f),
                new float3(1.77746f, -1.66f, 0.0707307f),
                new float3(2.13f, -1.77f, -1.62f),
                new float3(1, 0, 0)
            };
            offsets = new[]{
                new float3(0.353333f,  0.458333f, -0.081667f),
                new float3(0.493000f,  0.532167f, -0.449167f),
                new float3(0.551667f, -1.031667f, -0.255000f),
                new float3(0.235000f,  0.036667f,  0.128333f),
                new float3(0.346667f,  0.236667f,  0.321667f),
                new float3(0.638333f,  0.323333f,  0.181667f),
                new float3(0.258333f,  0.021667f,  0.420000f),
                new float3(0.595000f, -0.021500f, -0.491667f),
                new float3(0.484167f, -0.127500f,  0.694167f),
                new float3(0.385000f, -0.187167f, -0.260000f),
                new float3(0.756667f,  0.210000f, -0.016667f),
                new float3(0.333333f,  0.068333f,  0.238333f),
                new float3(1.238333f, -0.993333f,  1.038333f),
                new float3(0.206333f,  0.255500f, -0.180833f),
                new float3(0.245000f, -0.283333f,  0.066667f),
                new float3(0.206333f,  0.255500f, -0.180833f),
                new float3(0.182317f,  0.072492f,  0.518550f),
                new float3(1.125000f,  0.500000f,  0.000000f),
                new float3(0.573333f,  0.115000f,  0.190000f),
                new float3(0.418833f,  0.901117f,  0.418333f),
                new float3(0.800637f,  0.683333f,  0.231772f),
                new float3(0.643105f,  0.856235f,  0.153051f),
                new float3(0.781117f,  0.140627f, -0.330263f),
                new float3(0.831667f,  0.508333f,  0.746667f),
                new float3(0.000000f,  0.000000f,  0.000000f),
            };
            typeof(SplitterPanel).GetProperty("DoubleBuffered", BindingFlags.NonPublic | 
                BindingFlags.Instance).SetValue(ScreenDivider.Panel2, true, null);
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
            b = new Bitmap(width, height, width * bytes,
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
            p = TranslateSpace(p, shift);
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
                    return WarpDist(DE(p, side), i - 1, seed.x) * 6;
                }
            }
            return WarpDist(DE(p, side), iterations - 1, seed.x) * 6;
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
        private void OnLoad(object sender, EventArgs e)
        {
            Init();
            ScreenDivider.Panel2.Invalidate();
        }
        private void OnMouseDown(object sender, MouseEventArgs e)
        {
            mouseLocation = new int2(e.X, e.Y);
            isMouseDown = true;
        }
        private void OnMouseUp(object sender, MouseEventArgs e)
        {
            isMouseDown = false;
        }
        [GpuManaged]
        private void OnMouseMove(object sender, MouseEventArgs e)
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
                ScreenDivider.Panel2.Invalidate();
            }
            mouseLocation = new int2(e.X, e.Y);
        }
        private void OnMouseWheel(object sender, MouseEventArgs e)
        {
            if (mouseLocation.x >= 0 && mouseLocation.y >= 0 && mouseLocation.x < width * granularity && mouseLocation.y < height * granularity)
            {
                camera = A(camera, M(directions[Index(mouseLocation.x / granularity, mouseLocation.y / granularity, width)], movementSize * e.Delta / 100));
                ScreenDivider.Panel2.Invalidate();
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
        private void IndexChanged(object sender, EventArgs e)
        {
            seed = seeds[LevelDropdown.SelectedIndex];
            shift = offsets[LevelDropdown.SelectedIndex];
            SetScaleControls();
            SetAngle1Controls();
            SetAngle2Controls();
            SetOffsetControls();
            ScreenDivider.Panel2.Invalidate();
        }

        private void OnDraw(object sender, PaintEventArgs e)
        {
            if (directions.Length != 0)
            {
                MarchRays();
                e.Graphics.DrawImage(b, 0, 0, width * granularity, height * granularity);
            }
        }
        private void ScaleSlider_Scroll(object sender, EventArgs e)
        {
            seed.x = ScaleSlider.Value * 2.0f / ScaleSlider.Maximum;
            SetScaleControls();
            ScreenDivider.Panel2.Invalidate();
        }
        private void ScaleText_TextChanged(object sender, EventArgs e)
        {
            if (!float.TryParse(ScaleText.Text, out float scaleparsed))
            {
                SetScaleControls();
                return;
            }
            seed.x = scaleparsed;
            SetScaleControls();
            ScreenDivider.Panel2.Invalidate();
        }
        private void Angle1Slider_Scroll(object sender, EventArgs e)
        {
            seed.y = (float)(Angle1Slider.Value * Math.PI / Angle1Slider.Maximum);
            SetAngle1Controls();
            ScreenDivider.Panel2.Invalidate();
        }
        private void Angle1Text_TextChanged(object sender, EventArgs e)
        {
            if (!float.TryParse(Angle1Text.Text, out float angle1parsed))
            {
                SetAngle1Controls();
                return;
            }
            seed.y = angle1parsed;
            SetAngle1Controls();
            ScreenDivider.Panel2.Invalidate();
        }
        private void Angle2Slider_Scroll(object sender, EventArgs e)
        {
            seed.z = (float)(Angle2Slider.Value * Math.PI / Angle2Slider.Maximum);
            SetAngle2Controls();
            ScreenDivider.Panel2.Invalidate();
        }
        private void Angle2Text_TextChanged(object sender, EventArgs e)
        {
            if (!float.TryParse(Angle2Text.Text, out float angle2parsed))
            {
                SetAngle2Controls();
                return;
            }
            seed.y = angle2parsed;
            SetAngle2Controls();
            ScreenDivider.Panel2.Invalidate();
        }
        private void OffsetXText_TextChanged(object sender, EventArgs e)
        {
            if (!float.TryParse(OffsetXText.Text, out float offsetxparsed))
            {
                SetOffsetControls();
                return;
            }
            shift.x = offsetxparsed;
            SetOffsetControls();
            ScreenDivider.Panel2.Invalidate();
        }
        private void OffsetYText_TextChanged(object sender, EventArgs e)
        {
            if (!float.TryParse(OffsetYText.Text, out float offsetyparsed))
            {
                SetOffsetControls();
                return;
            }
            shift.y = offsetyparsed;
            SetOffsetControls();
            ScreenDivider.Panel2.Invalidate();
        }
        private void OffsetZText_TextChanged(object sender, EventArgs e)
        {
            if (!float.TryParse(OffsetZText.Text, out float offsetzparsed))
            {
                SetOffsetControls();
                return;
            }
            shift.z = offsetzparsed;
            SetOffsetControls();
            ScreenDivider.Panel2.Invalidate();
        }
        private void IterationsSlider_Scroll(object sender, EventArgs e)
        {
            iterations = IterationsSlider.Value;
            ScreenDivider.Panel2.Invalidate();
        }
        private void SetScaleControls()
        {
            ScaleText.Text = seed.x.ToString("0.000");
            ScaleSlider.Value = Math.Max(ScaleSlider.Minimum, Math.Min(ScaleSlider.Maximum, (int)(ScaleSlider.Maximum * seed.x / 2)));
        }
        private void SetAngle1Controls()
        {
            Angle1Text.Text = seed.y.ToString("0.00");
            Angle1Slider.Value = Math.Max(Angle1Slider.Minimum, Math.Min(Angle1Slider.Maximum, (int)(Angle1Slider.Maximum / Math.PI * seed.y)));
        }
        private void SetAngle2Controls()
        {
            Angle2Text.Text = seed.z.ToString("0.00");
            Angle2Slider.Value = Math.Max(Angle2Slider.Minimum, Math.Min(Angle2Slider.Maximum, (int)(Angle2Slider.Maximum / Math.PI * seed.z)));
        }
        private void SetOffsetControls()
        {
            OffsetXText.Text = shift.x.ToString("0.000");
            OffsetYText.Text = shift.y.ToString("0.000");
            OffsetZText.Text = shift.z.ToString("0.000");
        }
    }
}