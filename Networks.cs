using System;
using System.Collections.Generic;
using System.Linq;

namespace lab2_perceptron
{
    // Нейрон Розенблатта: y = step(w*x + b). Правило Хебба: dw = eta * x * t, t in {-1,+1} (при ошибке распознавания).
    // Loss - критерий персептрона, нормированный на длину вектора весов: среднее max(0, -t*net) / ||w||
    // (средняя геометрическая глубина ошибочно классифицированных образцов; плавно убывает при обучении).
    // History[0] - значение до обучения, History[k] - после k-й эпохи.
    // Лучшие веса (минимум числа ошибок) запоминаются и восстанавливаются после обучения (pocket).
    public class Perceptron
    {
        public double[] W; public double B;
        public List<double> History = new List<double>();

        public Perceptron(int nInputs, double[] init, Random rnd)
        {
            W = new double[nInputs];
            if (init != null && init.Length == nInputs + 1)
            { Array.Copy(init, W, nInputs); B = init[nInputs]; }
            else
            { for (int i = 0; i < nInputs; i++) W[i] = (rnd.NextDouble() - 0.5) * 0.02; B = (rnd.NextDouble() - 0.5) * 0.02; }
        }

        public double Net(double[] x)
        {
            double s = B;
            for (int i = 0; i < W.Length; i++) if (x[i] != 0) s += W[i] * x[i];
            return s;
        }
        public int Predict(double[] x) { return Net(x) >= 0 ? 1 : 0; }

        double Loss(double[][] X, int[] t01, out int errors)
        {
            double sum = 0; errors = 0;
            for (int i = 0; i < X.Length; i++)
            {
                double net = Net(X[i]);
                int t = 2 * t01[i] - 1;
                if ((net >= 0 ? 1 : -1) != t) errors++;
                sum += Math.Max(0, -t * net);
            }
            double norm = Math.Sqrt(W.Sum(v => v * v) + B * B) + 1e-9;
            return sum / X.Length / norm;
        }

        public void Fit(double[][] X, int[] t01, int epochs, double eta, Action<int, double> cb, Random rnd)
        {
            History.Clear();
            int n = X.Length, err;
            History.Add(Loss(X, t01, out err));
            double[] bestW = (double[])W.Clone(); double bestB = B; int bestErr = err;
            for (int ep = 1; ep <= epochs; ep++)
            {
                foreach (int i in Enumerable.Range(0, n).OrderBy(_ => rnd.Next()).ToArray())
                {
                    int t = 2 * t01[i] - 1;
                    int o = Net(X[i]) >= 0 ? 1 : -1;
                    if (o != t)
                    {
                        for (int j = 0; j < W.Length; j++) if (X[i][j] != 0) W[j] += eta * t * X[i][j];
                        B += eta * t;
                    }
                }
                double loss = Loss(X, t01, out err);
                History.Add(loss);
                if (err < bestErr) { bestErr = err; bestW = (double[])W.Clone(); bestB = B; }
                if (cb != null) cb(ep, loss);
                if (err == 0) break;
            }
            W = bestW; B = bestB;
        }
    }

    // Однослойная сеть: по одному нейрону на класс, решение - argmax.
    // Обучение (правило Хебба для многих классов): при ошибке веса нейрона верного класса усиливаются (+eta*x),
    // а веса ошибочно победившего нейрона ослабляются (-eta*x).
    // Loss - многоклассовый критерий персептрона: средняя нормированная глубина ошибки (net_победителя - net_верного)/||w_победителя - w_верного||.
    public class SingleLayerNet
    {
        public double[][] W; public double[] B; public int K, N;
        public List<double> History = new List<double>();

        public SingleLayerNet(int nInputs, int nClasses, double[][] init, Random rnd)
        {
            N = nInputs; K = nClasses;
            W = new double[K][]; B = new double[K];
            for (int k = 0; k < K; k++)
            {
                W[k] = new double[N];
                if (init != null && init.Length == K && init[k].Length == N + 1)
                { Array.Copy(init[k], W[k], N); B[k] = init[k][N]; }
                else
                { for (int i = 0; i < N; i++) W[k][i] = (rnd.NextDouble() - 0.5) * 0.02; B[k] = (rnd.NextDouble() - 0.5) * 0.02; }
            }
        }

        double Net(int k, double[] x)
        {
            double s = B[k];
            for (int i = 0; i < N; i++) if (x[i] != 0) s += W[k][i] * x[i];
            return s;
        }

        public int Predict(double[] x)
        {
            int best = 0; double bv = double.MinValue;
            for (int k = 0; k < K; k++) { double v = Net(k, x); if (v > bv) { bv = v; best = k; } }
            return best;
        }
        public int[] PredictAll(double[][] X) { return X.Select(x => Predict(x)).ToArray(); }

        double Loss(double[][] X, int[] y, out int errors)
        {
            double sum = 0; errors = 0;
            for (int i = 0; i < X.Length; i++)
            {
                int p = Predict(X[i]), t = y[i];
                if (p == t) continue;
                errors++;
                double dist = B[p] - B[t], sq = dist * dist;
                for (int j = 0; j < N; j++) { double d = W[p][j] - W[t][j]; sq += d * d; }
                sum += (Net(p, X[i]) - Net(t, X[i])) / (Math.Sqrt(sq) + 1e-9);
            }
            return sum / X.Length;
        }

        public void Fit(double[][] X, int[] y, int epochs, double eta, Action<int, double> cb, Random rnd)
        {
            History.Clear();
            int n = X.Length, err;
            History.Add(Loss(X, y, out err));
            double[][] bestW = W.Select(r => (double[])r.Clone()).ToArray(); double[] bestB = (double[])B.Clone();
            int bestErr = err;
            for (int ep = 1; ep <= epochs; ep++)
            {
                foreach (int i in Enumerable.Range(0, n).OrderBy(_ => rnd.Next()).ToArray())
                {
                    int p = Predict(X[i]);
                    int t = y[i];
                    if (p != t)
                    {
                        for (int j = 0; j < N; j++)
                            if (X[i][j] != 0) { W[t][j] += eta * X[i][j]; W[p][j] -= eta * X[i][j]; }
                        B[t] += eta; B[p] -= eta;
                    }
                }
                double loss = Loss(X, y, out err);
                History.Add(loss);
                if (err < bestErr) { bestErr = err; bestW = W.Select(r => (double[])r.Clone()).ToArray(); bestB = (double[])B.Clone(); }
                if (cb != null) cb(ep, loss);
                if (err == 0) break;
            }
            W = bestW; B = bestB;
        }

        public double[][] Weights()
        {
            return Enumerable.Range(0, K).Select(k => W[k].Concat(new[] { B[k] }).ToArray()).ToArray();
        }
    }

    // Многослойный персептрон: вход -> скрытый слой (tanh) -> softmax; backprop + мини-батч SGD, loss = cross-entropy.
    public class Mlp
    {
        int nIn, nH, nOut;
        double[,] W1, W2; double[] b1, b2;
        public List<double> History = new List<double>();

        static double Gauss(Random r)
        {
            double u1 = 1.0 - r.NextDouble(), u2 = r.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }

        public Mlp(int nIn, int nH, int nOut, Random rnd)
        {
            this.nIn = nIn; this.nH = nH; this.nOut = nOut;
            W1 = new double[nIn, nH]; W2 = new double[nH, nOut]; b1 = new double[nH]; b2 = new double[nOut];
            double s1 = Math.Sqrt(2.0 / nIn), s2 = Math.Sqrt(2.0 / nH);
            for (int i = 0; i < nIn; i++) for (int h = 0; h < nH; h++) W1[i, h] = Gauss(rnd) * s1;
            for (int h = 0; h < nH; h++) for (int k = 0; k < nOut; k++) W2[h, k] = Gauss(rnd) * s2;
        }

        void Forward(double[] x, double[] a1, double[] p)
        {
            for (int h = 0; h < nH; h++) a1[h] = b1[h];
            for (int i = 0; i < nIn; i++)
            {
                if (x[i] == 0) continue;
                for (int h = 0; h < nH; h++) a1[h] += x[i] * W1[i, h];
            }
            for (int h = 0; h < nH; h++) a1[h] = Math.Tanh(a1[h]);
            double mx = double.MinValue;
            for (int k = 0; k < nOut; k++)
            {
                double s = b2[k];
                for (int h = 0; h < nH; h++) s += a1[h] * W2[h, k];
                p[k] = s; if (s > mx) mx = s;
            }
            double sum = 0;
            for (int k = 0; k < nOut; k++) { p[k] = Math.Exp(p[k] - mx); sum += p[k]; }
            for (int k = 0; k < nOut; k++) p[k] /= sum;
        }

        public int Predict(double[] x)
        {
            double[] a1 = new double[nH], p = new double[nOut];
            Forward(x, a1, p);
            int best = 0; for (int k = 1; k < nOut; k++) if (p[k] > p[best]) best = k;
            return best;
        }
        public int[] PredictAll(double[][] X) { return X.Select(x => Predict(x)).ToArray(); }

        public void Fit(double[][] X, int[] y, int epochs, double lr, int batch, Action<int, double> cb, Random rnd)
        {
            History.Clear();
            int n = X.Length;
            double[] a1 = new double[nH], p = new double[nOut], d1 = new double[nH], d2 = new double[nOut];
            double[,] gW1 = new double[nIn, nH], gW2 = new double[nH, nOut];
            double[] gb1 = new double[nH], gb2 = new double[nOut];
            double l0 = 0;
            for (int i = 0; i < n; i++) { Forward(X[i], a1, p); l0 -= Math.Log(p[y[i]] + 1e-12); }
            History.Add(l0 / n);
            for (int ep = 1; ep <= epochs; ep++)
            {
                int[] order = Enumerable.Range(0, n).OrderBy(_ => rnd.Next()).ToArray();
                for (int s = 0; s < n; s += batch)
                {
                    int e = Math.Min(n, s + batch);
                    Array.Clear(gW1, 0, gW1.Length); Array.Clear(gW2, 0, gW2.Length);
                    Array.Clear(gb1, 0, nH); Array.Clear(gb2, 0, nOut);
                    for (int q = s; q < e; q++)
                    {
                        int i = order[q]; double[] x = X[i];
                        Forward(x, a1, p);
                        for (int k = 0; k < nOut; k++) d2[k] = p[k] - (y[i] == k ? 1 : 0);
                        for (int h = 0; h < nH; h++)
                        {
                            double sum = 0;
                            for (int k = 0; k < nOut; k++) { gW2[h, k] += a1[h] * d2[k]; sum += W2[h, k] * d2[k]; }
                            d1[h] = sum * (1 - a1[h] * a1[h]);
                            gb1[h] += d1[h];
                        }
                        for (int k = 0; k < nOut; k++) gb2[k] += d2[k];
                        for (int j = 0; j < nIn; j++)
                        {
                            if (x[j] == 0) continue;
                            for (int h = 0; h < nH; h++) gW1[j, h] += x[j] * d1[h];
                        }
                    }
                    double sc = lr / (e - s);
                    for (int j = 0; j < nIn; j++) for (int h = 0; h < nH; h++) W1[j, h] -= sc * gW1[j, h];
                    for (int h = 0; h < nH; h++) { b1[h] -= sc * gb1[h]; for (int k = 0; k < nOut; k++) W2[h, k] -= sc * gW2[h, k]; }
                    for (int k = 0; k < nOut; k++) b2[k] -= sc * gb2[k];
                }
                double loss = 0;
                for (int i = 0; i < n; i++) { Forward(X[i], a1, p); loss -= Math.Log(p[y[i]] + 1e-12); }
                loss /= n;
                History.Add(loss);
                if (cb != null) cb(ep, loss);
            }
        }
    }

    public static class XorDemo
    {
        public static void Run(out double accNeuron, out double accMlp, out List<double> h1, out List<double> h2)
        {
            double[][] X = { new double[] { 0, 0 }, new double[] { 0, 1 }, new double[] { 1, 0 }, new double[] { 1, 1 } };
            int[] t = { 0, 1, 1, 0 };
            Random rnd = new Random(1);
            Perceptron p = new Perceptron(2, null, rnd);
            p.Fit(X, t, 200, 0.1, null, rnd);
            int ok1 = 0; for (int i = 0; i < 4; i++) if (p.Predict(X[i]) == t[i]) ok1++;
            accNeuron = ok1 / 4.0;
            Mlp m = new Mlp(2, 8, 2, rnd);
            m.Fit(X, t, 3000, 0.5, 4, null, rnd);
            int ok2 = 0; for (int i = 0; i < 4; i++) if (m.Predict(X[i]) == t[i]) ok2++;
            accMlp = ok2 / 4.0;
            h1 = p.History; h2 = m.History;
        }
    }
}
