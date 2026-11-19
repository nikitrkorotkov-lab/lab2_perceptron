using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace lab2_perceptron
{
    // Параметры предобработки изображения.
    public class PrepOptions
    {
        public int Threshold = 0;   // 0 = автоматический порог (Otsu), 1..254 = ручной порог яркости
        public bool Crop = true;    // обрезать по объекту и масштабировать с сохранением пропорций
    }

    public class DataSet
    {
        public double[][] X;
        public int[] Y;
        public List<string> Names = new List<string>();
        public int Width, Height;
        public int Count { get { return X.Length; } }
        public int Features { get { return X[0].Length; } }

        static readonly string[] Ext = { ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff" };

        // Изображение -> бинарный вектор (объект = 1, фон = 0) независимо от того, светлый фон или тёмный.
        public static double[] ImageToVector(string path, int w, int h, PrepOptions o)
        {
            using (Bitmap src = new Bitmap(path))
            {
                int sw = src.Width, sh = src.Height;
                double k = Math.Min(1.0, 160.0 / Math.Max(sw, sh));
                int ww = Math.Max(1, (int)Math.Round(sw * k)), wh = Math.Max(1, (int)Math.Round(sh * k));
                using (Bitmap work = new Bitmap(ww, wh, PixelFormat.Format32bppArgb))
                {
                    using (Graphics g = Graphics.FromImage(work))
                    {
                        g.Clear(Color.Transparent);
                        g.CompositingMode = CompositingMode.SourceCopy;
                        g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                        g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                        g.DrawImage(src, new Rectangle(0, 0, ww, wh));
                    }
                    return Binarize(work, w, h, o);
                }
            }
        }

        static int Otsu(double[] gray)
        {
            int[] hist = new int[256];
            foreach (double g in gray) hist[Math.Max(0, Math.Min(255, (int)g))]++;
            double total = gray.Length, sum = 0;
            for (int i = 0; i < 256; i++) sum += i * hist[i];
            double sumB = 0, wB = 0, best = -1; int thr = 128;
            for (int t = 0; t < 256; t++)
            {
                wB += hist[t];
                if (wB == 0) continue;
                double wF = total - wB;
                if (wF == 0) break;
                sumB += t * hist[t];
                double mB = sumB / wB, mF = (sum - sumB) / wF;
                double between = wB * wF * (mB - mF) * (mB - mF);
                if (between > best) { best = between; thr = t; }
            }
            return thr;
        }

        static double[] Binarize(Bitmap b, int w, int h, PrepOptions o)
        {
            int W = b.Width, H = b.Height;
            double[] gray = new double[W * H];
            int[] alpha = new int[W * H];
            int minA = 255;
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                {
                    Color c = b.GetPixel(x, y);
                    int i = y * W + x;
                    gray[i] = 0.299 * c.R + 0.587 * c.G + 0.114 * c.B;
                    alpha[i] = c.A;
                    if (c.A < minA) minA = c.A;
                }
            bool[] mask = new bool[W * H];
            if (minA < 250)
            {
                // изображение с прозрачностью: объект = непрозрачные пиксели
                for (int i = 0; i < mask.Length; i++) mask[i] = alpha[i] >= 128;
            }
            else
            {
                int t = (o != null && o.Threshold > 0) ? o.Threshold : Otsu(gray);
                int bright = 0, total = 0;
                for (int x = 0; x < W; x++)
                {
                    total += 2;
                    if (gray[x] > t) bright++;
                    if (gray[(H - 1) * W + x] > t) bright++;
                }
                for (int y = 1; y < H - 1; y++)
                {
                    total += 2;
                    if (gray[y * W] > t) bright++;
                    if (gray[y * W + W - 1] > t) bright++;
                }
                bool bgBright = bright * 2 > total;   // фон определяется по рамке изображения
                for (int i = 0; i < mask.Length; i++) mask[i] = bgBright ? gray[i] <= t : gray[i] > t;
            }

            int minx = W, miny = H, maxx = -1, maxy = -1;
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    if (mask[y * W + x])
                    {
                        if (x < minx) minx = x; if (x > maxx) maxx = x;
                        if (y < miny) miny = y; if (y > maxy) maxy = y;
                    }
            double[] v = new double[w * h];
            if (maxx < 0) return v;

            bool crop = o == null || o.Crop;
            if (!crop) { minx = 0; miny = 0; maxx = W - 1; maxy = H - 1; }
            double bw = maxx - minx + 1, bh = maxy - miny + 1;
            double sx, sy, offX, offY;
            if (crop)
            {
                double s = Math.Min(Math.Max(1.0, w - 2.0) / bw, Math.Max(1.0, h - 2.0) / bh);
                sx = s; sy = s;
                offX = (w - bw * s) / 2.0; offY = (h - bh * s) / 2.0;
            }
            else
            {
                sx = w / bw; sy = h / bh; offX = 0; offY = 0;
            }
            for (int ty = 0; ty < h; ty++)
                for (int tx = 0; tx < w; tx++)
                {
                    double fx0 = minx + (tx - offX) / sx, fx1 = minx + (tx + 1 - offX) / sx;
                    double fy0 = miny + (ty - offY) / sy, fy1 = miny + (ty + 1 - offY) / sy;
                    int xa = Math.Max(0, (int)Math.Floor(fx0)), xb = Math.Min(W - 1, (int)Math.Ceiling(fx1) - 1);
                    int ya = Math.Max(0, (int)Math.Floor(fy0)), yb = Math.Min(H - 1, (int)Math.Ceiling(fy1) - 1);
                    if (xb < xa) xb = xa; if (yb < ya) yb = ya;
                    if (xa > W - 1 || ya > H - 1) continue;
                    int cnt = 0, all = 0;
                    for (int yy = ya; yy <= yb; yy++)
                        for (int xx = xa; xx <= xb; xx++) { all++; if (mask[yy * W + xx]) cnt++; }
                    if (all > 0 && cnt >= 0.3 * all) v[ty * w + tx] = 1.0;
                }
            return v;
        }

        public static DataSet LoadFolder(string root, int w, int h, PrepOptions o)
        {
            List<double[]> xs = new List<double[]>();
            List<int> ys = new List<int>();
            DataSet ds = new DataSet();
            foreach (string dir in Directory.GetDirectories(root).OrderBy(d => d, StringComparer.OrdinalIgnoreCase))
            {
                string cls = Path.GetFileName(dir);
                int idx = -1;
                foreach (string f in Directory.GetFiles(dir).OrderBy(d => d, StringComparer.OrdinalIgnoreCase))
                {
                    if (!Ext.Contains(Path.GetExtension(f).ToLowerInvariant())) continue;
                    if (idx < 0) { ds.Names.Add(cls); idx = ds.Names.Count - 1; }
                    xs.Add(ImageToVector(f, w, h, o));
                    ys.Add(idx);
                }
            }
            if (xs.Count == 0) throw new Exception("В папке нет подпапок с изображениями.");
            ds.X = xs.ToArray(); ds.Y = ys.ToArray(); ds.Width = w; ds.Height = h;
            return ds;
        }

        public void SaveCsv(string path)
        {
            using (StreamWriter sw = new StreamWriter(path, false, new UTF8Encoding(false)))
            {
                StringBuilder head = new StringBuilder();
                for (int i = 0; i < Features; i++) head.Append("p" + i + ";");
                head.Append("class");
                sw.WriteLine(head.ToString());
                for (int n = 0; n < Count; n++)
                {
                    StringBuilder sb = new StringBuilder();
                    for (int i = 0; i < Features; i++) sb.Append(X[n][i] != 0 ? "1;" : "0;");
                    sb.Append(Names[Y[n]]);
                    sw.WriteLine(sb.ToString());
                }
            }
        }

        public static DataSet LoadCsv(string path)
        {
            string[] lines = File.ReadAllLines(path).Where(l => l.Trim().Length > 0).ToArray();
            if (lines.Length == 0) throw new Exception("Пустой CSV.");
            char sep = lines[0].IndexOf(';') >= 0 ? ';' : ',';
            int start = 0;
            double tmp;
            if (!double.TryParse(lines[0].Split(sep)[0].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out tmp)) start = 1;
            List<double[]> xs = new List<double[]>();
            List<string> labels = new List<string>();
            for (int i = start; i < lines.Length; i++)
            {
                string[] p = lines[i].Split(sep);
                int n = p.Length - 1;
                double[] v = new double[n];
                for (int j = 0; j < n; j++)
                    v[j] = double.Parse(p[j].Trim(), CultureInfo.InvariantCulture) != 0 ? 1.0 : 0.0;
                xs.Add(v);
                labels.Add(p[n].Trim());
            }
            DataSet ds = new DataSet();
            List<string> uniq = labels.Distinct().OrderBy(s => s, StringComparer.OrdinalIgnoreCase).ToList();
            ds.Names = uniq;
            ds.X = xs.ToArray();
            ds.Y = labels.Select(l => uniq.IndexOf(l)).ToArray();
            int side = (int)Math.Round(Math.Sqrt(ds.Features));
            ds.Width = side; ds.Height = ds.Features / Math.Max(side, 1);
            if (ds.Width * ds.Height != ds.Features) { ds.Width = ds.Features; ds.Height = 1; }
            return ds;
        }

        public void Split(out int[] train, out int[] test)
        {
            Random rnd = new Random(42);
            List<int> tr = new List<int>(), te = new List<int>();
            for (int c = 0; c < Names.Count; c++)
            {
                List<int> idx = Enumerable.Range(0, Count).Where(i => Y[i] == c).OrderBy(i => rnd.Next()).ToList();
                int k = (int)Math.Round(0.8 * idx.Count);
                if (idx.Count > 1) k = Math.Min(Math.Max(k, 1), idx.Count - 1);
                tr.AddRange(idx.Take(k)); te.AddRange(idx.Skip(k));
            }
            train = tr.OrderBy(i => rnd.Next()).ToArray();
            test = te.OrderBy(i => rnd.Next()).ToArray();
        }

        public double[][] Sub(int[] idx) { return idx.Select(i => X[i]).ToArray(); }
        public int[] SubY(int[] idx) { return idx.Select(i => Y[i]).ToArray(); }

        public static void CreateRandomWeights(string path, int nInputs, int nNeurons)
        {
            Random rnd = new Random();
            double[][] W = new double[nNeurons][];
            for (int k = 0; k < nNeurons; k++)
            {
                W[k] = new double[nInputs + 1];
                for (int i = 0; i <= nInputs; i++) W[k][i] = (rnd.NextDouble() - 0.5) * 0.02;
            }
            SaveWeights(path, W);
        }

        public static void SaveWeights(string path, double[][] W)
        {
            File.WriteAllLines(path, W.Select(r => string.Join(";", r.Select(v => v.ToString("F6", CultureInfo.InvariantCulture)))));
        }

        public static double[][] LoadWeights(string path)
        {
            return File.ReadAllLines(path).Where(l => l.Trim().Length > 0)
                .Select(l => l.Split(';').Select(s => double.Parse(s.Trim(), CultureInfo.InvariantCulture)).ToArray()).ToArray();
        }
    }
}
