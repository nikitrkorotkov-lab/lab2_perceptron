using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Windows.Forms;

namespace lab2_perceptron
{
    // Отдельная форма с графиком loss-функции и сохранением в файл.
    // Точка с индексом k - значение loss после k-й эпохи (k = 0 - до обучения).
    public class LossForm : Form
    {
        Panel chart; Button btnSave;
        Dictionary<string, List<double>> curves = new Dictionary<string, List<double>>();
        static readonly Color[] Pal = { Color.RoyalBlue, Color.Crimson, Color.ForestGreen, Color.DarkOrange, Color.Purple };

        class BufferedPanel : Panel { public BufferedPanel() { DoubleBuffered = true; } }

        public LossForm()
        {
            Text = "График loss-функции"; Width = 760; Height = 560; StartPosition = FormStartPosition.CenterParent;
            btnSave = new Button { Text = "Сохранить график (PNG)", Dock = DockStyle.Bottom, Height = 34 };
            btnSave.Click += (s, e) => SaveImage();
            chart = new BufferedPanel { Dock = DockStyle.Fill, BackColor = Color.White };
            chart.Paint += (s, e) => Render(e.Graphics, chart.Width, chart.Height);
            chart.Resize += (s, e) => chart.Invalidate();
            Controls.Add(chart); Controls.Add(btnSave);
            FormClosing += (s, e) => { if (e.CloseReason == CloseReason.UserClosing) { e.Cancel = true; Hide(); } };
        }

        public void ShowCurves(Dictionary<string, List<double>> c)
        {
            curves = c.ToDictionary(k => k.Key, k => new List<double>(k.Value));
            chart.Invalidate();
            if (!Visible) Show();
            BringToFront();
        }

        void Render(Graphics g, int w, int h)
        {
            g.Clear(Color.White);
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
            int L = 70, R = 20, T = 40, B = 60;
            Rectangle pr = new Rectangle(L, T, Math.Max(10, w - L - R), Math.Max(10, h - T - B));
            using (Font f = new Font("Arial", 9))
            using (Font fb = new Font("Arial", 11, FontStyle.Bold))
            {
                g.DrawString("Изменение loss-функции в ходе обучения", fb, Brushes.Black, L, 8);
                g.DrawRectangle(Pens.Black, pr);
                if (curves.Count == 0) return;
                int maxPts = Math.Max(2, curves.Values.Max(v => v.Count));
                double maxV = curves.Values.SelectMany(v => v).DefaultIfEmpty(1).Max();
                if (maxV <= 0) maxV = 1;
                for (int i = 0; i <= 5; i++)
                {
                    int y = pr.Bottom - i * pr.Height / 5;
                    using (Pen gp = new Pen(Color.LightGray)) g.DrawLine(gp, pr.Left, y, pr.Right, y);
                    g.DrawString((maxV * i / 5).ToString("0.###"), f, Brushes.Black, 4, y - 7);
                    int x = pr.Left + i * pr.Width / 5;
                    g.DrawLine(Pens.Black, x, pr.Bottom, x, pr.Bottom + 4);
                    g.DrawString(((int)Math.Round((maxPts - 1) * i / 5.0)).ToString(), f, Brushes.Black, x - 8, pr.Bottom + 6);
                }
                g.DrawString("Эпоха", f, Brushes.Black, pr.Left + pr.Width / 2 - 15, pr.Bottom + 26);
                g.DrawString("Loss", f, Brushes.Black, 8, 22);
                int ci = 0, ly = T + 6;
                foreach (KeyValuePair<string, List<double>> kv in curves)
                {
                    Color col = Pal[ci % Pal.Length];
                    List<double> v = kv.Value;
                    if (v.Count > 1)
                    {
                        PointF[] pts = new PointF[v.Count];
                        for (int i = 0; i < v.Count; i++)
                            pts[i] = new PointF(pr.Left + (float)((double)i / (maxPts - 1) * pr.Width),
                                                pr.Bottom - (float)(v[i] / maxV * pr.Height));
                        using (Pen p = new Pen(col, 2)) g.DrawLines(p, pts);
                    }
                    else if (v.Count == 1)
                    {
                        using (Brush b = new SolidBrush(col)) g.FillEllipse(b, pr.Left - 3, pr.Bottom - (float)(v[0] / maxV * pr.Height) - 3, 6, 6);
                    }
                    using (Brush b = new SolidBrush(col)) g.FillRectangle(b, pr.Right - 230, ly + 3, 14, 8);
                    g.DrawString(kv.Key, f, Brushes.Black, pr.Right - 210, ly);
                    ly += 18; ci++;
                }
            }
        }

        void SaveImage()
        {
            using (SaveFileDialog d = new SaveFileDialog { Filter = "PNG|*.png|JPEG|*.jpg|BMP|*.bmp", FileName = "loss.png" })
            {
                if (d.ShowDialog() != DialogResult.OK) return;
                using (Bitmap bmp = new Bitmap(1200, 800))
                {
                    using (Graphics g = Graphics.FromImage(bmp)) Render(g, bmp.Width, bmp.Height);
                    ImageFormat fmt = d.FilterIndex == 2 ? ImageFormat.Jpeg : d.FilterIndex == 3 ? ImageFormat.Bmp : ImageFormat.Png;
                    bmp.Save(d.FileName, fmt);
                }
            }
        }
    }
}
