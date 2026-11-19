using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace lab2_perceptron
{
    public partial class Form1 : Form
    {
        DataSet ds; int[] tr, te;
        Perceptron neuron; int neuronCls = -1;
        SingleLayerNet layer; Mlp mlp;
        string initWeightsPath;
        LossForm lossForm;
        Random rnd = new Random(1);

        Label lblInfo; TextBox tbW, tbH, tbThr, log; ProgressBar pb;
        ComboBox cbClass, cbAct;
        TextBox nEta, nEp, lEta, lEp, mEta, mEp, mHid;
        PictureBox pbN, pbL, pbM; Label resN, resL, resM;
        TabControl tabs;

        public Form1()
        {
            InitializeComponent();
            Text = "ЛР №2 - Однослойная нейронная сеть (персептрон Розенблатта, правило Хебба)";
            Width = 1100; Height = 800; StartPosition = FormStartPosition.CenterScreen;
            BuildUi();
        }

        static Label Lbl(Control p, string t, int x, int y)
        { Label l = new Label { Text = t, Left = x, Top = y + 3, AutoSize = true }; p.Controls.Add(l); return l; }
        static TextBox Tb(Control p, string t, int x, int y, int w)
        { TextBox b = new TextBox { Text = t, Left = x, Top = y, Width = w }; p.Controls.Add(b); return b; }
        static Button Btn(Control p, string t, int x, int y, int w, EventHandler h)
        { Button b = new Button { Text = t, Left = x, Top = y, Width = w, Height = 28 }; b.Click += h; p.Controls.Add(b); return b; }
        static PictureBox Pic(Control p, int x, int y)
        { PictureBox b = new PictureBox { Left = x, Top = y, Width = 164, Height = 164, BorderStyle = BorderStyle.FixedSingle, BackColor = Color.White }; p.Controls.Add(b); return b; }

        void BuildUi()
        {
            GroupBox top = new GroupBox { Text = "1. Датасет и начальные веса", Dock = DockStyle.Top, Height = 100 };
            Btn(top, "Папка с изображениями...", 10, 20, 170, OpenFolder);
            Btn(top, "Загрузить CSV...", 185, 20, 120, OpenCsv);
            Btn(top, "Сохранить датасет в CSV", 310, 20, 160, SaveCsv);
            Lbl(top, "Размер (Ш x В):", 490, 20);
            tbW = Tb(top, "32", 585, 20, 35); tbH = Tb(top, "32", 625, 20, 35);
            Lbl(top, "Порог (0 = авто, версия Оцу):", 680, 20);
            tbThr = Tb(top, "0", 865, 20, 40);
            Btn(top, "Создать файл случайных весов...", 10, 55, 210, MakeWeights);
            Btn(top, "Загрузить файл весов...", 225, 55, 170, PickWeights);
            lblInfo = new Label { Text = "Датасет не загружен", Left = 405, Top = 60, AutoSize = true };
            top.Controls.Add(lblInfo);

            tabs = new TabControl { Dock = DockStyle.Top, Height = 230 };
            TabPage t1 = new TabPage("Нейрон (верно / не верно)"), t2 = new TabPage("Однослойная сеть"),
                    t3 = new TabPage("Многослойный персептрон"), t4 = new TabPage("Линейная несепарабельность (XOR)");
            tabs.TabPages.AddRange(new TabPage[] { t1, t2, t3, t4 });

            Lbl(t1, "Распознаваемый класс:", 10, 15);
            cbClass = new ComboBox { Left = 160, Top = 12, Width = 150, DropDownStyle = ComboBoxStyle.DropDownList }; t1.Controls.Add(cbClass);
            Lbl(t1, "Скорость обучения:", 10, 50); nEta = Tb(t1, "0.1", 160, 47, 60);
            Lbl(t1, "Эпох:", 240, 50); nEp = Tb(t1, "100", 290, 47, 60);
            Btn(t1, "Обучить нейрон (правило Хебба)", 10, 85, 220, TrainNeuron);
            Btn(t1, "Проверить изображение...", 240, 85, 180, CheckNeuron);
            resN = new Label { Left = 10, Top = 130, AutoSize = true, Font = new Font("Arial", 14, FontStyle.Bold) }; t1.Controls.Add(resN);
            pbN = Pic(t1, 600, 10);

            Lbl(t2, "Скорость обучения:", 10, 15); lEta = Tb(t2, "0.1", 160, 12, 60);
            Lbl(t2, "Эпох:", 240, 15); lEp = Tb(t2, "100", 290, 12, 60);
            Btn(t2, "Обучить слой нейронов", 10, 50, 200, TrainLayer);
            Btn(t2, "Распознать изображение...", 220, 50, 190, (s, e) => CheckNet(false));
            Btn(t2, "Сохранить веса...", 420, 50, 140, SaveLayerWeights);
            resL = new Label { Left = 10, Top = 100, AutoSize = true, Font = new Font("Arial", 14, FontStyle.Bold) }; t2.Controls.Add(resL);
            pbL = Pic(t2, 600, 10);

            Lbl(t3, "Скорость обучения:", 10, 15); mEta = Tb(t3, "0.05", 160, 12, 60);
            Lbl(t3, "Эпох:", 240, 15); mEp = Tb(t3, "200", 290, 12, 60);
            Lbl(t3, "Нейронов в скрытом слое:", 10, 50); mHid = Tb(t3, "128", 190, 47, 60);
            Lbl(t3, "Активация:", 270, 50);
            cbAct = new ComboBox { Left = 345, Top = 47, Width = 80, DropDownStyle = ComboBoxStyle.DropDownList };
            cbAct.Items.AddRange(new object[] { "tanh", "relu", "sigmoid" }); cbAct.SelectedIndex = 0; t3.Controls.Add(cbAct);
            Btn(t3, "Обучить MLP", 10, 85, 160, TrainMlp);
            Btn(t3, "Распознать изображение...", 180, 85, 190, (s, e) => CheckNet(true));
            resM = new Label { Left = 10, Top = 130, AutoSize = true, Font = new Font("Arial", 14, FontStyle.Bold) }; t3.Controls.Add(resM);
            pbM = Pic(t3, 600, 10);

            Lbl(t4, "Один нейрон не решает XOR (классы линейно несепарабельны), многослойный персептрон - решает.", 10, 15);
            Btn(t4, "Запустить демонстрацию XOR", 10, 50, 220, RunXor);

            log = new TextBox { Multiline = true, ScrollBars = ScrollBars.Both, ReadOnly = true, Dock = DockStyle.Fill, Font = new Font("Consolas", 9), WordWrap = false };
            pb = new ProgressBar { Dock = DockStyle.Bottom, Height = 18 };

            Controls.Add(log); Controls.Add(pb); Controls.Add(tabs); Controls.Add(top);
            log.BringToFront();
        }

        void Log(string s) { log.AppendText(s + "\r\n"); }
        static double D(TextBox t) { return double.Parse(t.Text.Trim().Replace(',', '.'), CultureInfo.InvariantCulture); }
        static int I(TextBox t) { return int.Parse(t.Text.Trim()); }

        int Thr()
        {
            int v;
            return int.TryParse(tbThr.Text.Trim(), out v) && v > 0 ? v : 0;
        }

        bool NeedData()
        {
            if (ds == null) { MessageBox.Show("Сначала загрузите датасет.", "Нет данных"); return false; }
            return true;
        }

        void SetData(DataSet d)
        {
            ds = d; ds.Split(out tr, out te);
            neuron = null; layer = null; mlp = null;
            cbClass.Items.Clear(); foreach (string n in ds.Names) cbClass.Items.Add(n);
            cbClass.SelectedIndex = 0;
            string cnt = string.Join(", ", ds.Names.Select((n, i) => n + "=" + ds.Y.Count(y => y == i)));
            lblInfo.Text = "Образцов: " + ds.Count + ", признаков: " + ds.Features + ", классов: " + ds.Names.Count
                + " | обучающая: " + tr.Length + ", тестовая: " + te.Length + " (80/20)";
            Log("Датасет загружен. Классы: " + cnt);
        }

        void Fail(Exception ex) { MessageBox.Show(ex.Message, "Ошибка", MessageBoxButtons.OK, MessageBoxIcon.Error); }

        void OpenFolder(object s, EventArgs e)
        {
            using (FolderBrowserDialog d = new FolderBrowserDialog { Description = "Папка датасета (каждая подпапка - класс)" })
            {
                if (d.ShowDialog() != DialogResult.OK) return;
                try { Cursor = Cursors.WaitCursor; SetData(DataSet.LoadFolder(d.SelectedPath, I(tbW), I(tbH), Thr())); }
                catch (Exception ex) { Fail(ex); }
                finally { Cursor = Cursors.Default; }
            }
        }

        void OpenCsv(object s, EventArgs e)
        {
            using (OpenFileDialog d = new OpenFileDialog { Filter = "CSV|*.csv|Все файлы|*.*" })
            {
                if (d.ShowDialog() != DialogResult.OK) return;
                try { SetData(DataSet.LoadCsv(d.FileName)); tbW.Text = ds.Width.ToString(); tbH.Text = ds.Height.ToString(); }
                catch (Exception ex) { Fail(ex); }
            }
        }

        void SaveCsv(object s, EventArgs e)
        {
            if (!NeedData()) return;
            using (SaveFileDialog d = new SaveFileDialog { Filter = "CSV|*.csv", FileName = "dataset.csv" })
                if (d.ShowDialog() == DialogResult.OK) { ds.SaveCsv(d.FileName); Log("CSV сохранён: " + d.FileName); }
        }

        void MakeWeights(object s, EventArgs e)
        {
            if (!NeedData()) return;
            using (SaveFileDialog d = new SaveFileDialog { Filter = "CSV|*.csv", FileName = "random_weights.csv" })
                if (d.ShowDialog() == DialogResult.OK)
                {
                    DataSet.CreateRandomWeights(d.FileName, ds.Features, ds.Names.Count); initWeightsPath = d.FileName;
                    Log("Создан файл случайных весов (" + ds.Names.Count + " x " + (ds.Features + 1) + "): " + d.FileName);
                }
        }

        void PickWeights(object s, EventArgs e)
        {
            using (OpenFileDialog d = new OpenFileDialog { Filter = "CSV/TXT|*.csv;*.txt" })
                if (d.ShowDialog() == DialogResult.OK) { initWeightsPath = d.FileName; Log("Начальные веса: " + d.FileName); }
        }

        double[][] InitWeights()
        {
            if (initWeightsPath == null || !File.Exists(initWeightsPath)) return null;
            double[][] W = DataSet.LoadWeights(initWeightsPath);
            if (W.Length == ds.Names.Count && W[0].Length == ds.Features + 1) return W;
            Log("Файл весов не подходит по размеру - используются случайные веса.");
            return null;
        }

        void ShowLoss(Dictionary<string, List<double>> c)
        {
            if (lossForm == null || lossForm.IsDisposed) lossForm = new LossForm();
            lossForm.ShowCurves(c);
        }

        void Progress(int ep, int total)
        {
            BeginInvoke(new Action(() => { pb.Value = Math.Max(0, Math.Min(100, ep * 100 / total)); }));
        }

        void Busy(bool b) { tabs.Enabled = !b; Cursor = b ? Cursors.WaitCursor : Cursors.Default; if (!b) pb.Value = 0; }

        void Preview(PictureBox box, double[] v)
        {
            int w = ds.Width, h = ds.Height;
            if (w * h != v.Length) { w = (int)Math.Round(Math.Sqrt(v.Length)); h = v.Length / w; }
            int sc = Math.Max(1, 160 / Math.Max(w, h));
            Bitmap bmp = new Bitmap(w * sc, h * sc);
            using (Graphics g = Graphics.FromImage(bmp))
            {
                g.Clear(Color.White);
                for (int y = 0; y < h; y++) for (int x = 0; x < w; x++)
                        if (v[y * w + x] != 0) g.FillRectangle(Brushes.Black, x * sc, y * sc, sc, sc);
            }
            box.SizeMode = PictureBoxSizeMode.CenterImage; box.Image = bmp;
        }

        double[] PickImage()
        {
            using (OpenFileDialog d = new OpenFileDialog { Filter = "Изображения|*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.tif" })
            {
                if (d.ShowDialog() != DialogResult.OK) return null;
                return DataSet.ImageToVector(d.FileName, ds.Width, ds.Height, Thr());
            }
        }

        string ReportBoth(string title, Func<double[][], int[]> predict, out double minTestAcc)
        {
            double a, b;
            string s1 = Metrics.Report(title + ": обучающая выборка", ds.SubY(tr), predict(ds.Sub(tr)), ds.Names, out a);
            string s2 = Metrics.Report(title + ": тестовая выборка", ds.SubY(te), predict(ds.Sub(te)), ds.Names, out b);
            minTestAcc = b;
            return s1 + "\r\n" + s2;
        }

        async void TrainNeuron(object s, EventArgs e)
        {
            if (!NeedData()) return;
            double eta; int ep;
            try { eta = D(nEta); ep = I(nEp); } catch { MessageBox.Show("Некорректные параметры."); return; }
            int c = cbClass.SelectedIndex; neuronCls = c;
            double[][] W = InitWeights();
            neuron = new Perceptron(ds.Features, W != null ? W[c] : null, rnd);
            double[][] Xtr = ds.Sub(tr); int[] ttr = ds.SubY(tr).Select(y => y == c ? 1 : 0).ToArray();
            Perceptron nn = neuron;
            Busy(true);
            try { await Task.Run(() => nn.Fit(Xtr, ttr, ep, eta, (i, l) => Progress(i, ep), rnd)); }
            catch (Exception ex) { Fail(ex); Busy(false); return; }
            Busy(false);
            ShowLoss(new Dictionary<string, List<double>> { { "Нейрон \"" + ds.Names[c] + "\"", nn.History } });
            Log("Нейрон \"" + ds.Names[c] + "\" обучен за " + nn.History.Count + " эп., loss=" + nn.History.Last().ToString("F4"));
            string[] titles = { "обучающая", "тестовая" };
            int[][] sets = { tr, te };
            for (int q = 0; q < 2; q++)
            {
                int[] t = ds.SubY(sets[q]).Select(y => y == c ? 1 : 0).ToArray();
                int[] p = ds.Sub(sets[q]).Select(x => nn.Predict(x)).ToArray();
                Metrics.ClassMetrics m = Metrics.Binary(t, p, 1);
                Log("  " + titles[q] + ": Accuracy=" + m.Acc.ToString("F3") + " Precision=" + m.Prec.ToString("F3")
                    + " Recall=" + m.Rec.ToString("F3") + " F1=" + m.F1.ToString("F3")
                    + " (TP=" + m.TP + " TN=" + m.TN + " FP=" + m.FP + " FN=" + m.FN + ")");
            }
        }

        void CheckNeuron(object s, EventArgs e)
        {
            if (!NeedData()) return;
            if (neuron == null) { MessageBox.Show("Сначала обучите нейрон."); return; }
            double[] v = PickImage(); if (v == null) return;
            Preview(pbN, v);
            bool ok = neuron.Predict(v) == 1;
            resN.ForeColor = ok ? Color.Green : Color.Red;
            resN.Text = (ok ? "TRUE: это " : "FALSE: это не ") + "\"" + ds.Names[neuronCls] + "\"";
        }

        async void TrainLayer(object s, EventArgs e)
        {
            if (!NeedData()) return;
            double eta; int ep;
            try { eta = D(lEta); ep = I(lEp); } catch { MessageBox.Show("Некорректные параметры."); return; }
            layer = new SingleLayerNet(ds.Features, ds.Names.Count, InitWeights(), rnd);
            double[][] Xtr = ds.Sub(tr); int[] ytr = ds.SubY(tr);
            SingleLayerNet net = layer;
            Busy(true);
            try { await Task.Run(() => net.Fit(Xtr, ytr, ep, eta, (i, l) => Progress(i, ep), rnd)); }
            catch (Exception ex) { Fail(ex); Busy(false); return; }
            Busy(false);
            ShowLoss(new Dictionary<string, List<double>> { { "Однослойная сеть (MSE)", net.History } });
            Log("Слой обучен за " + net.History.Count + " эп., loss=" + net.History.Last().ToString("F4"));
            double mt; Log(ReportBoth("Однослойная сеть", net.PredictAll, out mt));
        }

        void SaveLayerWeights(object s, EventArgs e)
        {
            if (layer == null) { MessageBox.Show("Сеть не обучена."); return; }
            using (SaveFileDialog d = new SaveFileDialog { Filter = "CSV|*.csv", FileName = "trained_weights.csv" })
                if (d.ShowDialog() == DialogResult.OK) { DataSet.SaveWeights(d.FileName, layer.Weights()); Log("Веса сохранены: " + d.FileName); }
        }

        async void TrainMlp(object s, EventArgs e)
        {
            if (!NeedData()) return;
            double eta; int ep, hid;
            try { eta = D(mEta); ep = I(mEp); hid = I(mHid); } catch { MessageBox.Show("Некорректные параметры."); return; }
            mlp = new Mlp(ds.Features, hid, ds.Names.Count, cbAct.SelectedItem.ToString(), rnd);
            double[][] Xtr = ds.Sub(tr); int[] ytr = ds.SubY(tr);
            Mlp m = mlp;
            Busy(true);
            try { await Task.Run(() => m.Fit(Xtr, ytr, ep, eta, 16, (i, l) => Progress(i, ep), rnd)); }
            catch (Exception ex) { Fail(ex); Busy(false); return; }
            Busy(false);
            ShowLoss(new Dictionary<string, List<double>> { { "MLP (cross-entropy)", m.History } });
            Log("MLP обучен, loss=" + m.History.Last().ToString("F4"));
            double mt; Log(ReportBoth("MLP", m.PredictAll, out mt));
            Log("Требование Accuracy >= 0.95 для каждого класса (тест): " + (mt >= 0.95 ? "ВЫПОЛНЕНО" : "НЕ выполнено - увеличьте число нейронов/эпох"));
        }

        void CheckNet(bool isMlp)
        {
            if (!NeedData()) return;
            if ((isMlp && mlp == null) || (!isMlp && layer == null)) { MessageBox.Show("Сначала обучите сеть."); return; }
            double[] v = PickImage(); if (v == null) return;
            int cls = isMlp ? mlp.Predict(v) : layer.Predict(v);
            Preview(isMlp ? pbM : pbL, v);
            Label target = isMlp ? resM : resL;
            target.Text = "Класс: " + ds.Names[cls];
        }

        void RunXor(object s, EventArgs e)
        {
            double a1, a2; List<double> h1, h2;
            XorDemo.Run(out a1, out a2, out h1, out h2);
            Log("XOR: один нейрон Accuracy=" + a1.ToString("F2") + " (максимум 0.75), MLP Accuracy=" + a2.ToString("F2"));
            ShowLoss(new Dictionary<string, List<double>> { { "Нейрон (MSE)", h1 }, { "MLP (cross-entropy)", h2 } });
        }
    }
}
