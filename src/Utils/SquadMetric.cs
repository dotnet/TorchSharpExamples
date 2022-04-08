// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;

namespace Examples.Utils
{
    public static class SquadMetric
    {
        public static double CosineSimilarity(IList<double> vector1, IList<double> vector2)
        {
            if (vector1.Count != vector2.Count)
            {
                throw new ArgumentException(
                    $"Vectors to compute cosine similarity must have the same length, got {vector1.Count} and {vector2.Count}");
            }

            var xMax = vector1.Max();
            var yMax = vector2.Max();
            var xEnum = vector1.Select(x => x / xMax);
            var yEnum = vector2.Select(y => y / yMax);
            var nominator = Enumerable.Zip(xEnum, yEnum).Sum(pair => pair.First * pair.Second);
            var denominator = Math.Sqrt(xEnum.Sum(x => x * x) * yEnum.Sum(y => y * y));
            return nominator / denominator;
        }

        public static double ComputeF1(long predictionStart, long predictionEnd, long truthStart, long truthEnd)
        {
            // No answer
            if ((truthStart | truthEnd) == 0)
            {
                if ((predictionStart | predictionEnd) == 0) return 1.0;
                else return 0.0;
            }
            // Has answer: prediction invalid
            if (predictionStart > predictionEnd)
            {
                return 0.0;
            }
            // Has answer: prediction valid
            var prediction = predictionEnd - predictionStart + 1;
            var truth = truthEnd - truthStart + 1;
            var overlap = ComputeOverlap(predictionStart, predictionEnd, truthStart, truthEnd);
            if (overlap == 0)
            {
                return 0.0;
            }
            var precision = 1.0 * overlap / prediction;
            var recall = 1.0 * overlap / truth;
            var f1 = 2 * precision * recall / (precision + recall);
            return f1;
        }

        private static long ComputeOverlap(long predictionStart, long predictionEnd, long truthStart, long truthEnd)
        {
            var arr = new long[] { predictionStart, predictionEnd, truthStart, truthEnd };
            var max = arr.Max();
            var min = arr.Min();
            //var overlap = (truthEnd - truthStart + 1) + (predictionEnd - predictionStart + 1) - (max - min + 1);
            var overlap = (truthEnd - truthStart) + (predictionEnd - predictionStart) - (max - min) + 1;
            return Math.Max(overlap, 0);
        }

        public static IList<(int start, int end, double score)> ComputeTopKSpansWithScore(
            torch.Tensor predictStartScores, torch.Tensor predictStarts, torch.Tensor predictEndScores, torch.Tensor predictEnds, int k)
        {
            var startScores = predictStartScores.ToArray<float>();
            var endScores = predictEndScores.ToArray<float>();
            var starts = predictStarts.ToArray<long>();
            var ends = predictEnds.ToArray<long>();
            var topK = new List<(int, int, double)>();
            for (var i = 0; i < starts.Length; ++i)
            {
                for (var j = 0; j < ends.Length; ++j)
                {
                    if (starts[i] <= ends[j])
                    {
                        topK.Add(((int)starts[i], (int)ends[j], startScores[i] * endScores[j]));
                    }
                }
            }
            topK = topK.OrderByDescending(tuple => tuple.Item3).Take(k).ToList();
            return topK;
        }
    }
}