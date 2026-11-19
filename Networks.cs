using System;
using System.Collections.Generic;
using System.Linq;

namespace lab2_perceptron
{
    // Нейрон Розенблатта: y = step(w*x + b). Правило Хебба: dw = eta * x * t, t in {-1,+1} (при ошибке).
    // Для линейно несепарабельных данных используется pocket-стратегия: в конце возвращаются лучшие веса эпохи.
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
            { for (int i = 0; i < nInputs; i++) W[i] = rnd.NextDouble() - 0.5; B = rnd.NextDouble() - 0.5; }
        }

        public double Net(double[] x)
        {
            double s = B;
            for (int i = 0; i < W.Length; i++) if (x[i] != 0) s += W[i] * x[i];
            return s;
        }
        public int Predict(double[] x) { return Net(x) >= 0 ? 1 : 0; }

        public void Fit(double[][] X, int[] t01, int epochs, double eta, Action<int, double> cb, Random rnd)
        {
            History.Clear();
            int n = X.Length;
            double bestLoss = double.MaxValue, bestB = B;
            double[] bestW = (double[])W.Clone();
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
                double loss = 0;
                for (int i = 0; i < n; i++) { double d = t01[i] - Predict(X[i]); loss += d * d; }
                loss /= n;
                History.Add(loss);
                if (loss <= bestLoss) { bestLoss = loss; bestB = B; Array.Copy(W, bestW, W.Length); }
                if (cb != null) cb(ep, loss);
                if (loss == 0) break;
            }
            Array.Copy(bestW, W, W.Length); B = bestB;
        }
    }

    // Однослойная сеть: по одному нейрону на класс, решение - argmax. Pocket: остаются веса эпохи с минимальной ошибкой распознавания.
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
                { for (int i = 0; i < N; i++) W[k][i] = rnd.NextDouble() - 0.5; B[k] = rnd.NextDouble() - 0.5; }
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

        public void Fit(double[][] X, int[] y, int epochs, double eta, Action<int, double> cb, Random rnd)
        {
            History.Clear();
            int n = X.Length;
            double bestErr = double.MaxValue;
            double[][] bestW = new double[K][]; double[] bestB = new double[K];
            for (int k = 0; k < K; k++) bestW[k] = new double[N];
            for (int ep = 1; ep <= epochs; ep++)
            {
                foreach (int i in Enumerable.Range(0, n).OrderBy(_ => rnd.Next()).ToArray())
                    for (int k = 0; k < K; k++)
                    {
                        int t = (y[i] == k) ? 1 : -1;
                        int o = Net(k, X[i]) >= 0 ? 1 : -1;
                        if (o != t)
                        {
                            for (int j = 0; j < N; j++) if (X[i][j] != 0) W[k][j] += eta * t * X[i][j];
                            B[k] += eta * t;
                        }
                    }
                double loss = 0;
                for (int i = 0; i < n; i++)
                    for (int k = 0; k < K; k++)
                    {
                        double d = (y[i] == k ? 1 : 0) - (Net(k, X[i]) >= 0 ? 1 : 0);
                        loss += d * d;
                    }
                loss /= (n * K);
                int wrong = 0;
                for (int i = 0; i < n; i++) if (Predict(X[i]) != y[i]) wrong++;
                History.Add(loss);
                if (wrong <= bestErr)
                {
                    bestErr = wrong;
                    for (int k = 0; k < K; k++) { Array.Copy(W[k], bestW[k], N); bestB[k] = B[k]; }
                }
                if (cb != null) cb(ep, loss);
                if (loss == 0) break;
            }
            for (int k = 0; k < K; k++) { Array.Copy(bestW[k], W[k], N); B[k] = bestB[k]; }
        }

        public double[][] Weights()
        {
            return Enumerable.Range(0, K).Select(k => W[k].Concat(new[] { B[k] }).ToArray()).ToArray();
        }
    }

    // Многослойный персептрон: вход -> скрытый слой -> softmax; backprop + мини-батч SGD с momentum, loss = cross-entropy.
    public class Mlp
    {
        int nIn, nH, nOut; string act;
        double[,] W1, W2; double[] b1, b2;
        public List<double> History = new List<double>();

        static double Gauss(Random r)
        {
            double u1 = 1.0 - r.NextDouble(), u2 = r.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }

        public Mlp(int nIn, int nH, int nOut, string act, Random rnd)
        {
            this.nIn = nIn; this.nH = nH; this.nOut = nOut; this.act = act;
            W1 = new double[nIn, nH]; W2 = new double[nH, nOut]; b1 = new double[nH]; b2 = new double[nOut];
            double s1 = Math.Sqrt(2.0 / nIn), s2 = Math.Sqrt(2.0 / nH);
            for (int i = 0; i < nIn; i++) for (int h = 0; h < nH; h++) W1[i, h] = Gauss(rnd) * s1;
            for (int h = 0; h < nH; h++) for (int k = 0; k < nOut; k++) W2[h, k] = Gauss(rnd) * s2;
        }

        double A(double z)
        {
            if (act == "relu") return z > 0 ? z : 0;
            if (act == "sigmoid") return 1.0 / (1.0 + Math.Exp(-Math.Max(-50, Math.Min(50, z))));
            return Math.Tanh(z);
        }
        double dA(double a, double z)
        {
            if (act == "relu") return z > 0 ? 1 : 0;
            if (act == "sigmoid") return a * (1 - a);
            return 1 - a * a;
        }

        void Forward(double[] x, double[] z1, double[] a1, double[] p)
        {
            for (int h = 0; h < nH; h++) z1[h] = b1[h];
            for (int i = 0; i < nIn; i++)
            {
                if (x[i] == 0) continue;
                for (int h = 0; h < nH; h++) z1[h] += x[i] * W1[i, h];
            }
            for (int h = 0; h < nH; h++) a1[h] = A(z1[h]);
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
            double[] z1 = new double[nH], a1 = new double[nH], p = new double[nOut];
            Forward(x, z1, a1, p);
            int best = 0; for (int k = 1; k < nOut; k++) if (p[k] > p[best]) best = k;
            return best;
        }
        public int[] PredictAll(double[][] X) { return X.Select(x => Predict(x)).ToArray(); }

        public void Fit(double[][] X, int[] y, int epochs, double lr, int batch, Action<int, double> cb, Random rnd)
        {
            History.Clear();
            const double mom = 0.9;
            int n = X.Length;
            double[] z1 = new double[nH], a1 = new double[nH], p = new double[nOut], d1 = new double[nH], d2 = new double[nOut];
            double[,] gW1 = new double[nIn, nH], gW2 = new double[nH, nOut];
            double[] gb1 = new double[nH], gb2 = new double[nOut];
            double[,] vW1 = new double[nIn, nH], vW2 = new double[nH, nOut];
            double[] vb1 = new double[nH], vb2 = new double[nOut];
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
                        Forward(x, z1, a1, p);
                        for (int k = 0; k < nOut; k++) d2[k] = p[k] - (y[i] == k ? 1 : 0);
                        for (int h = 0; h < nH; h++)
                        {
                            double sum = 0;
                            for (int k = 0; k < nOut; k++) { gW2[h, k] += a1[h] * d2[k]; sum += W2[h, k] * d2[k]; }
                            d1[h] = sum * dA(a1[h], z1[h]);
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
                    for (int j = 0; j < nIn; j++)
                        for (int h = 0; h < nH; h++) { vW1[j, h] = mom * vW1[j, h] - sc * gW1[j, h]; W1[j, h] += vW1[j, h]; }
                    for (int h = 0; h < nH; h++)
                    {
                        vb1[h] = mom * vb1[h] - sc * gb1[h]; b1[h] += vb1[h];
                        for (int k = 0; k < nOut; k++) { vW2[h, k] = mom * vW2[h, k] - sc * gW2[h, k]; W2[h, k] += vW2[h, k]; }
                    }
                    for (int k = 0; k < nOut; k++) { vb2[k] = mom * vb2[k] - sc * gb2[k]; b2[k] += vb2[k]; }
                }
                double loss = 0;
                for (int i = 0; i < n; i++) { Forward(X[i], z1, a1, p); loss -= Math.Log(p[y[i]] + 1e-12); }
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
            Mlp m = new Mlp(2, 8, 2, "tanh", rnd);
            m.Fit(X, t, 3000, 0.1, 4, null, rnd);
            int ok2 = 0; for (int i = 0; i < 4; i++) if (m.Predict(X[i]) == t[i]) ok2++;
            accMlp = ok2 / 4.0;
            h1 = p.History; h2 = m.History;
        }
    }
}
