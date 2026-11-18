using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace lab2_perceptron
{
    public class DataSet
    {
        public double[][] X;
        public int[] Y;
        public List<string> Names = new List<string>();
        public int Width, Height;
        public int Count { get { return X.Length; } }
        public int Features { get { return X[0].Length; } }

        static readonly string[] Ext = { ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff" };

        public static double[] ImageToVector(string path, int w, int h, bool invert)
        {
            using (Bitmap src = new Bitmap(path))
            using (Bitmap bmp = new Bitmap(w, h))
            {
                using (Graphics g = Graphics.FromImage(bmp))
                {
                    g.InterpolationMode = InterpolationMode.NearestNeighbor;
                    g.PixelOffsetMode = PixelOffsetMode.Half;
                    g.DrawImage(src, new Rectangle(0, 0, w, h));
                }
                double[] v = new double[w * h];
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                    {
                        Color c = bmp.GetPixel(x, y);
                        int gray = (int)(0.299 * c.R + 0.587 * c.G + 0.114 * c.B);
                        bool on = invert ? gray < 255 : gray != 0;
                        v[y * w + x] = on ? 1.0 : 0.0;
                    }
                return v;
            }
        }

        public static DataSet LoadFolder(string root, int w, int h, bool invert)
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
                    xs.Add(ImageToVector(f, w, h, invert));
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
                for (int i = 0; i <= nInputs; i++) W[k][i] = rnd.NextDouble() - 0.5;
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
