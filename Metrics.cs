using System;
using System.Collections.Generic;
using System.Text;

namespace lab2_perceptron
{
    public static class Metrics
    {
        public class ClassMetrics { public double Acc, Prec, Rec, F1; public int TP, FP, FN, TN; }

        public static ClassMetrics Binary(int[] t, int[] p, int cls)
        {
            ClassMetrics m = new ClassMetrics();
            for (int i = 0; i < t.Length; i++)
            {
                bool tt = t[i] == cls, pp = p[i] == cls;
                if (tt && pp) m.TP++; else if (!tt && pp) m.FP++; else if (tt) m.FN++; else m.TN++;
            }
            int n = t.Length;
            m.Acc = n == 0 ? 0 : (double)(m.TP + m.TN) / n;
            m.Prec = m.TP + m.FP == 0 ? 0 : (double)m.TP / (m.TP + m.FP);
            m.Rec = m.TP + m.FN == 0 ? 0 : (double)m.TP / (m.TP + m.FN);
            m.F1 = m.Prec + m.Rec == 0 ? 0 : 2 * m.Prec * m.Rec / (m.Prec + m.Rec);
            return m;
        }

        public static string Report(string title, int[] t, int[] p, List<string> names, out double minClassAcc)
        {
            int K = names.Count; int[,] cm = new int[K, K];
            for (int i = 0; i < t.Length; i++) cm[t[i], p[i]]++;
            int ok = 0; for (int k = 0; k < K; k++) ok += cm[k, k];
            StringBuilder sb = new StringBuilder();
            sb.AppendLine("=== " + title + " (" + t.Length + " образцов) ===");
            sb.AppendLine("Общая Accuracy: " + ((double)ok / Math.Max(1, t.Length)).ToString("F4"));
            sb.AppendLine("Класс".PadRight(12) + "Accuracy".PadLeft(9) + "Precision".PadLeft(10) + "Recall".PadLeft(8) + "F1".PadLeft(8));
            minClassAcc = 1; double sp = 0, sr = 0, sf = 0;
            for (int k = 0; k < K; k++)
            {
                ClassMetrics m = Binary(t, p, k);
                minClassAcc = Math.Min(minClassAcc, m.Acc);
                sp += m.Prec; sr += m.Rec; sf += m.F1;
                sb.AppendLine(names[k].PadRight(12) + m.Acc.ToString("F3").PadLeft(9) + m.Prec.ToString("F3").PadLeft(10)
                    + m.Rec.ToString("F3").PadLeft(8) + m.F1.ToString("F3").PadLeft(8));
            }
            sb.AppendLine("macro: Precision=" + (sp / K).ToString("F3") + " Recall=" + (sr / K).ToString("F3") + " F1=" + (sf / K).ToString("F3"));
            sb.AppendLine("Матрица ошибок (строки - истина, столбцы - прогноз):");
            sb.Append("".PadRight(12));
            for (int k = 0; k < K; k++) sb.Append(Cut(names[k], 5).PadLeft(6));
            sb.AppendLine();
            for (int i = 0; i < K; i++)
            {
                sb.Append(Cut(names[i], 11).PadRight(12));
                for (int j = 0; j < K; j++) sb.Append(cm[i, j].ToString().PadLeft(6));
                sb.AppendLine();
            }
            return sb.ToString();
        }

        static string Cut(string s, int n) { return s.Length > n ? s.Substring(0, n) : s; }
    }
}
