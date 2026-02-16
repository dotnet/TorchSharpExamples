// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System.Collections.Generic;
using System.Linq;

namespace Examples.Utils
{
    public class RobertaInputBuilder
    {
        private readonly RobertaTokenizer Tokenizer;
        public readonly int MaxPositions;
        private readonly int[] Positions;
        private readonly int[] Zeros;
        private readonly int[] Ones;

        public RobertaInputBuilder(RobertaTokenizer tokenizer, int maxPositions)
        {
            Tokenizer = tokenizer;
            MaxPositions = maxPositions;
            Positions = Enumerable.Range(0, maxPositions).ToArray();
            Zeros = Enumerable.Repeat(0, maxPositions).ToArray();
            Ones = Enumerable.Repeat(1, maxPositions).ToArray();
        }

        /// <summary>
        /// Build Tensor input for the model according to the actual length of input tokens.
        /// [CLS] and [SEP] tokens will be added.
        /// </summary>
        /// <remarks>
        /// This method is only for input with one segment.
        /// </remarks>
        /// <param name="input">A list of tokenized input tokens.</param>
        /// <returns>A triplet of (token, position, token type).</returns>
        public (int[] tokens, int[] positions, int[] segments) Build(IList<string> tokens)
        {
            var tokenizedId = Tokenizer.TokenizeToId(tokens).ToList();
            tokenizedId.Insert(0, Tokenizer.BosIndex);
            tokenizedId.Add(Tokenizer.EosIndex);
            return (
                tokenizedId.ToArray(),
                Positions[0..tokenizedId.Count],
                Zeros[0..tokenizedId.Count]);
        }

        /// <summary>
        /// Build Tensor input for the model according to the actual length of input tokens.
        /// [CLS] and [SEP] tokens will be added.
        /// </summary>
        /// <remarks>
        /// This method is only for input with two segments.
        /// </remarks>
        /// <param name="input">A list of tokenized input tokens.</param>
        /// <returns>A triplet of (token, position, token type).</returns>
        public (int[] tokens, int[] positions, int[] segments) Build(IList<string> tokens0, IList<string> tokens1)
        {
            var tokenizedId0 = Tokenizer.TokenizeToId(tokens0);
            var tokenizedId1 = Tokenizer.TokenizeToId(tokens1);
            var tokenizedId = tokenizedId0.ToList();
            tokenizedId.Insert(0, Tokenizer.BosIndex);
            tokenizedId.Add(Tokenizer.EosIndex);
            tokenizedId.AddRange(tokenizedId1);
            tokenizedId = tokenizedId.Take(MaxPositions - 1).ToList();
            tokenizedId.Add(Tokenizer.EosIndex);
            var segments = Zeros[0..(tokenizedId0.Count + 2)]
                .Concat(Ones[0..(tokenizedId.Count - tokenizedId0.Count - 3)]).ToArray();
            return (
                tokenizedId.ToArray(),
                Positions[0..tokenizedId.Count],
                segments);
        }

        /// <summary>
        /// Build Tensor input for the model according to the actual length of input tokens.
        /// [CLS] and [SEP] tokens will be added.
        /// </summary>
        /// <remarks>
        /// This method is only for input with one segment.
        /// </remarks>
        /// <param name="input">A list of tokenized input token IDs.</param>
        /// <returns>A triplet of (token, position, token type).</returns>
        public (int[] tokens, int[] positions, int[] segments) Build(IList<int> tokensId)
        {
            tokensId.Insert(0, Tokenizer.BosIndex);
            tokensId.Add(Tokenizer.EosIndex);
            return (
                tokensId.ToArray(),
                Positions[0..tokensId.Count],
                Zeros[0..tokensId.Count]);
        }

        /// <summary>
        /// Build Tensor input for the model according to the actual length of input tokens.
        /// [CLS] and [SEP] tokens will be added.
        /// </summary>
        /// <remarks>
        /// This method is only for input with two segments.
        /// </remarks>
        /// <param name="input">A list of tokenized input token IDs.</param>
        /// <returns>A triplet of (token, position, token type).</returns>
        public (int[] tokens, int[] positions, int[] segments, int questionSegmentLength)
            Build(IList<int> tokensId0, IList<int> tokensId1)
        {
            var tokensId = tokensId0.ToList();
            tokensId.Insert(0, Tokenizer.BosIndex);
            tokensId.Add(Tokenizer.EosIndex);
            tokensId.AddRange(tokensId1);
            tokensId = tokensId.Take(MaxPositions - 1).ToList();
            tokensId.Add(Tokenizer.EosIndex);
            var segments = Zeros[0..(tokensId0.Count + 2)]
                .Concat(Ones[0..(tokensId.Count - tokensId0.Count - 2)]).ToArray();
            return (
                tokensId.ToArray(),
                Positions[0..tokensId.Count],
                segments,
                tokensId0.Count + 2);
        }

        /// <summary>
        /// Build Tensor input for the model according to the actual length of input tokens.
        /// [CLS] and [SEP] tokens will be added.
        /// </summary>
        /// <remarks>
        /// This method is only for input with two segments.
        /// </remarks>
        /// <param name="input">A list of tokenized input token IDs.</param>
        /// <returns>A tuple of (token, position, token type, startPosition, endPosition).</returns>
        public (int[] tokens, int[] positions, int[] segments, int[] startPositions, int[] endPositions, int questionSegmentLength)
            Build(IList<int> tokensId0, IList<int> tokensId1, IList<(int, int)> answerPositions)
        {
            var (tokens, positions, segments, questionSegmentLength) = Build(tokensId0, tokensId1);

            // There are only one answer in training set.
            // Filter out samples with the answer in truncated part.
            var groundTruths = new HashSet<(int, int)>();
            foreach (var answer in answerPositions)
            {
                var groundStart = answer.Item1 + questionSegmentLength;
                var groundEnd = answer.Item2 + questionSegmentLength;
                if (groundEnd < MaxPositions - 1)
                {
                    groundTruths.Add((groundStart, groundEnd));
                }
            }
            if (groundTruths.Count == 0)
            {
                groundTruths.Add((0, 0));
            }
            var startPositions = groundTruths.Select(p => p.Item1).ToArray();
            var endPositions = groundTruths.Select(p => p.Item2).ToArray();

            return (tokens, positions, segments, startPositions, endPositions, questionSegmentLength);
        }
    }
}