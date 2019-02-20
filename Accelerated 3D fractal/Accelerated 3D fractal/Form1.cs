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
        private ColorPalette cp;
        private float3 camera;
        private float cameraBaseDist;
        private float3 center;
        private float3 lightLocation;
        private float3 x;
        private float3 y;
        private float3 z;
        private Timer t;
        private float elapsed = 0;
        private float baseMovementSize = 0.5f;
        private float movementSize;
        private float scale = 3.1f;
        private float diffuse = 0.5f;
        private float ambient = 0.5f;
        private float focalLength = -1;
        private float minDist = 1;
        private float maxDist = 5;
        private int maxStep = 10000;
        private int granularity = 1;
        private int iterations = 8;
        private float side = 1;
        private int Width;
        private int Height;
        private float ambientOccStrength = 0.03f;
        public float shadowStrength = 8;
        private readonly dim3 BlockSize = new dim3(32, 32);
        private dim3 GridSize;
        private LaunchParam launchParam;
        public Gpu gpu;
        public Form1()
        {
            InitializeComponent();
            gpu = Gpu.Default;
            DoubleBuffered = true;
        }
        public void Init()
        {
            scale = 1 / scale;
            Width = ClientRectangle.Width / granularity / 2 * 2;
            Height = ClientRectangle.Height / granularity / 2 * 2;
            directions = new float3[Width * Height];
            dirDevMem = gpu.AllocateDevice(directions);
            dirDevPtr = dirDevMem.Ptr;
            pixels = new byte[Width * Height];
            pixDevMem = gpu.AllocateDevice(pixels);
            pixDevPtr = pixDevMem.Ptr;
            b = new Bitmap(Width, Height, Width,
                     PixelFormat.Format8bppIndexed,
                     Marshal.UnsafeAddrOfPinnedArrayElement(pixels, 0));
            center = new float3(0, 0, 0);
            camera = new float3(0, 0, 1.5f);
            cameraBaseDist = 1.5f;
            lightLocation = new float3(-2, 4, 6);
            GridSize = new dim3(Width / BlockSize.x, Height / BlockSize.y);
            launchParam = new LaunchParam(GridSize, BlockSize);
            minDist = ScaledDE(camera, iterations, scale, side) / Height;
            movementSize = minDist * Height * baseMovementSize;
            GetDirections();
            cp = b.Palette;
            for(int i = 0; i < 256; i++)
            {
                cp.Entries[i] = Color.FromArgb(i, i, i);
            }
            b.Palette = cp;
            x = new float3(1, 0, 0);
            y = new float3(0, 1, 0);
            z = new float3(0, 0, -1);
            t = new Timer();
            t.Interval = 1;
            t.Enabled = false;
            t.Tick += Update;
        }
        private void Update(object sender, EventArgs e)
        {
            elapsed += 0.001f;
            camera = new float3((float)(Math.Sin(elapsed) * cameraBaseDist), 0, (float)(Math.Cos(elapsed) * cameraBaseDist));
            Invalidate();
        }

        public static float l(float3 p)
        {
            return DeviceFunction.Sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        }
        public static float o(float3 a, float3 b)
        {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }
        public static float3 d(float3 p, float d)
        {
            p.x /= d;
            p.y /= d;
            p.z /= d;
            return p;
        }
        public static float3 m(float3 p, float d)
        {
            p.x *= d;
            p.y *= d;
            p.z *= d;
            return p;
        }
        public static float3 a(float3 a, float3 b)
        {
            a.x += b.x;
            a.y += b.y;
            a.z += b.z;
            return a;
        }
        public static float3 s(float3 a, float3 b)
        {
            a.x -= b.x;
            a.y -= b.y;
            a.z -= b.z;
            return a;
        }
        public static float3 r(float3 a, float3 b)
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
            float3 p = new float3((i - width / 2) / height, (j - height / 2) / height, focalLength);
            p = d(p, l(p));
            directions[h] = p;
        }
        public void GetColor(deviceptr<Color> colors)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            colors[i] = Color.FromArgb(i, i, i);
        }
        public static void MarchRay(deviceptr<float3> directions, deviceptr<byte> pixelValues, float3 camera, float3 light, float diffuse,
            float ambient, float ambientOccStrength, float minDist, float maxDist, int maxStep, int width, int height, int iterations,
            float scale, float side, float shadowStrength)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;
            int h = Index(i, j, width);
            float3 p = camera;
            int stepnum = 0;
            float dist = minDist + 1;
            float totalDist = 0;
            while (totalDist < maxDist && dist > minDist && stepnum < maxStep)
            {
                dist = ScaledDE(p, iterations, scale, side);
                p = a(p, m(directions[h], dist));
                totalDist += dist;
                stepnum++;
            }
            float greyscale = 0;
            if (DeviceFunction.Abs(dist) <= minDist)
            {
                float3 off = s(light, p);
                float lightVectorLength = l(off);
                off = d(off, lightVectorLength);
                //float3 marchedPoint = MarchRay(lightLocation, off, iterations, scale, side, minDist, maxDist, maxStep);
                float shadow = 1;
                float diffuseCalculated = 0;
                float normalAngle = o(off, Normal(p, iterations, scale, side, minDist));
                if (normalAngle > 0)
                {
                    shadow = NewSoftShadow(p, off, shadowStrength, iterations, scale, side, minDist, lightVectorLength, 0.01f);
                    diffuseCalculated = DeviceFunction.Max(diffuse * shadow * normalAngle, 0);
                }
                greyscale = (diffuseCalculated + ambient / (1 + stepnum * ambientOccStrength));
                greyscale = DeviceFunction.Min(DeviceFunction.Max(greyscale, 0), 1);
            }
            pixelValues[h] = (byte)(greyscale * byte.MaxValue);
        }
        [GpuManaged]
        public void GetDirections()
        {
            gpu.Launch(GetDirection, launchParam, dirDevPtr, focalLength, (float)Width, (float)Height);
            Gpu.Copy(gpu, dirDevPtr, directions, 0L, Width * Height);
        }
        [GpuManaged]
        public void MarchRays()
        {
            if (cp == null)
                return;
            gpu.Launch(MarchRay, launchParam, dirDevPtr, pixDevPtr, camera, lightLocation, diffuse, ambient, ambientOccStrength, minDist, maxDist, maxStep,
                Width, Height, iterations, scale, side, shadowStrength);
            Gpu.Copy(gpu, pixDevPtr, pixels, 0L, Width * Height);
            b = new Bitmap(Width, Height, Width,
                     PixelFormat.Format8bppIndexed,
                     Marshal.UnsafeAddrOfPinnedArrayElement(pixels, 0));
            b.Palette = cp;
            //for (int i = 0; i < Width * Height; i++)
            //{
             //   int greyscale = (int)(pixels[i] * 255);
            //    b.SetPixel(i % Width, Height - 1 - i / Width, Color.FromArgb(greyscale, greyscale, greyscale));
           //}
        }
        public static float OldSoftShadow(float3 p, float3 d, float shadowStrength, int iterations, float scale, float side, float minDist, float maxDist, float minAngle)
        {
            float k = 1;
            float dist = minDist;
            float angle = 1;
            float totalDist = minDist / 100;
            float3 marchedPoint = p;
            while (totalDist < maxDist)
            {
                dist = ScaledDE(marchedPoint, iterations, scale, side);
                if (dist == 0)
                    dist = minDist;
                marchedPoint = a(marchedPoint, m(d, dist));
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
        public static float NewSoftShadow(float3 p, float3 d, float shadowStrength, int iterations, float scale, float side, float minDist, float maxDist, float minAngle)
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
                dist = ScaledDE(a(p, m(d, totalDist)), iterations, scale, side);
                if (dist == 0)
                    dist = 0.000000001f;
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
        public static float3 Normal(float3 p, int iterations, float scale, float side, float epsilon)
        {
            float3 scaled = new float3(
                ScaledDE(new float3(p.x + epsilon, p.y, p.z), iterations, scale, side) -
                ScaledDE(new float3(p.x - epsilon, p.y, p.z), iterations, scale, side),
                ScaledDE(new float3(p.x, p.y + epsilon, p.z), iterations, scale, side) -
                ScaledDE(new float3(p.x, p.y - epsilon, p.z), iterations, scale, side),
                ScaledDE(new float3(p.x, p.y, p.z + epsilon), iterations, scale, side) -
                ScaledDE(new float3(p.x, p.y, p.z - epsilon), iterations, scale, side));
            return d(scaled, l(scaled));
        }
        public static float DE(float3 p, float side)
        {
            return CubeDE(p, new float3(0, 0, 0), side);
        }
        public static float ScaledDE(float3 p, int iterations, float scale, float side)
        {
            return WarpDist(DE(WarpSpace(p, iterations, scale, side), side), iterations, scale);
        }
        public static float SphereDE(float3 p, float3 c, float di)
        {
            return l(s(p, c)) - di / 2;
        }
        public static float CubeDE(float3 p, float3 c, float di)
        {
            float3 o = s(p, c);
            float d = DeviceFunction.Max(DeviceFunction.Abs(o.x), DeviceFunction.Max(DeviceFunction.Abs(o.y), DeviceFunction.Abs(o.z)));
            return d - di / 2;
        }
        public static float3 WarpSpace(float3 p, int iterations, float scale, float side)
        {
            for (int i = 0; i < iterations; i++)
            {
                p = ScaleSpace(p, scale);
                p = AbsSpace(p);
                p = FoldSpace(p, new float3(1, -1, 0));
                p = FoldSpace(p, new float3(1, 0, -1));
                p = TranslateSpace(p, new float3(side * 0.2f, 0, 0));
                p = AbsSpaceX(p);
                p = TranslateSpace(p, new float3(side * 0.2f, 0, 0));
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
            return d * s;
        }
        public static float3 ScaleSpace(float3 p, float s)
        {
            return d(p, s);
        }
        public static float3 TranslateSpace(float3 p, float3 offset)
        {
            return s(p, offset);
        }
        public static float3 ModSpace(float3 p, float3 mod)
        {
            return s(r(a(r(a(p, d(mod, 2)), mod), mod), mod), d(mod, 2));
        }
        public static float3 FoldSpace(float3 p, float3 n)
        {
            if (o(p, n) >= 0)
                return p;
            else
                return s(p, d(m(m(n, 2), o(p, n)), o(n, n)));
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
            z.y = DeviceFunction.Cos(t) * z.y + DeviceFunction.Sin(t) * z.z;
            z.z = DeviceFunction.Cos(t) * z.z - DeviceFunction.Sin(t) * z.y;
            return z;
        }
        public static float3 RotateY(float3 z, float t)
        {
            z.x = DeviceFunction.Cos(t) * z.x - DeviceFunction.Sin(t) * z.z;
            z.z = DeviceFunction.Cos(t) * z.z + DeviceFunction.Sin(t) * z.x;
            return z;
        }
        public static float3 RotateZ(float3 z, float t)
        {
            z.x = DeviceFunction.Cos(t) * z.x + DeviceFunction.Sin(t) * z.y;
            z.y = DeviceFunction.Cos(t) * z.y - DeviceFunction.Sin(t) * z.x;
            return z;
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
                e.Graphics.DrawImage(b, Point.Empty);
            }
        }

        private void Form1_KeyDown(object sender, KeyEventArgs e)
        {
            switch (e.KeyCode)
            {
                case Keys.D:
                    camera = a(camera, m(x, movementSize));
                    break;
                case Keys.A:
                    camera = s(camera, m(x, movementSize));
                    break;
                case Keys.Space:
                    camera = a(camera, m(y, movementSize));
                    break;
                case Keys.ShiftKey:
                    camera = s(camera, m(y, movementSize));
                    break;
                case Keys.W:
                    camera = a(camera, m(z, movementSize));
                    break;
                case Keys.S:
                    camera = s(camera, m(z, movementSize));
                    break;
            }
            cameraBaseDist = l(camera);
            minDist = ScaledDE(camera, iterations, scale, side) / Height;
            movementSize = minDist * Height  * baseMovementSize;
            if (minDist <= 0)
                minDist = 1;
            Invalidate();
        }
    }
}