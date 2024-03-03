// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.Linq;

namespace Examples.Utils
{
    /// <summary>
    /// A class for selecting the best matched documents given a query using a variant of TF-IDF algorithm.
    /// Visit https://en.wikipedia.org/wiki/Tf%E2%80%93idf for more information about the algorithm.
    /// </summary>
    public class TfIdfDocumentSelector
    {
        /// <summary>
        /// Smoothing factor for max-normalization of tf.
        /// </summary>
        private const double MaxNormSmoothing = 0.01;

        /// <summary>
        /// Weight of title.
        /// </summary>
        private const double TitleWeight = 0.1;

        /// <summary>
        /// Consider some tokens as stop words and do not count them in cosine similarity:
        ///     the first N tokens of the highest frequency in the vocab.
        /// </summary>
        private const int StopWordFilterIndex = 500;

        /// <summary>
        /// Consider some tokens as stop words and do not count them in cosine similarity:
        ///     the tokens with document frequency >= RATIO * total # documents.
        /// </summary>
        private const double StopWordFilterRatio = 0.25;

        /// <summary>
        /// SquadDocument frequency of each token.
        /// </summary>
        private readonly int[] DocumentFrequency;

        private readonly SquadDocument[] Documents;
        private readonly int DocumentCount;
        private readonly int VocabularySize;

        public IReadOnlyDictionary<SquadDocument, double[]> DocumentVectors { get; }

        private readonly RobertaTokenizer Tokenizer;
        private readonly HashSet<int> StopWordsTokenIds;

        public TfIdfDocumentSelector(IEnumerable<SquadDocument> documents, RobertaTokenizer tokenizer, bool useTitle = true)
        {
            Documents = documents.ToArray();
            Tokenizer = tokenizer;
            VocabularySize = tokenizer.VocabSize;

            DocumentCount = Documents.Length;
            DocumentFrequency = new int[VocabularySize];
            foreach (var doc in Documents)
            {
                var contextTokens = Tokenizer.TokenizeToId(doc.Context.ToLower());
                foreach (var token in contextTokens.ToHashSet())
                {
                    ++DocumentFrequency[token];
                }
            }

            var stopWordsTokenIds = Enumerable.Range(0, StopWordFilterIndex);   // high frequency
            for (var i = 0; i < VocabularySize; ++i)
            {
                if (DocumentFrequency[i] >= StopWordFilterRatio * Documents.Length)
                {
                    stopWordsTokenIds = stopWordsTokenIds.Append(i);
                }
            }
            StopWordsTokenIds = stopWordsTokenIds.ToHashSet();

            if (useTitle)
            {
                var documentVectors = new Dictionary<SquadDocument, double[]>();
                foreach (var doc in Documents)
                {
                    var titleTokens = Tokenizer.TokenizeToId(doc.Title.ToLower());
                    var titleVector = TfIdf(titleTokens);
                    var contextTokens = Tokenizer.TokenizeToId(doc.Context.ToLower());
                    var contextVector = TfIdf(contextTokens, maxNormTf: true);
                    var combined = Enumerable.Zip(titleVector, contextVector)
                        .Select(pair => TitleWeight * pair.First + (1 - TitleWeight) * pair.Second)
                        .ToArray();
                    documentVectors[doc] = combined;
                }
                DocumentVectors = documentVectors;
            }
            else
            {
                DocumentVectors = Documents.ToDictionary(
                    doc => doc,
                    doc => TfIdf(Tokenizer.TokenizeToId(doc.Context.ToLower()), maxNormTf: true));
            }
        }

        public SquadDocument Top1(string question)
        {
            return TopK(question, 1)[0];
        }

        public IList<SquadDocument> TopK(string question, int k)
        {
            var questionTokens = Tokenizer.TokenizeToId(question.ToLower());
            var weight = TfIdf(questionTokens);
            var matchedDocuments = DocumentVectors
                .Select(pair => (Doc: pair.Key, Sim: CosineSimilarityOfQuestion(weight, pair.Value)))
                .OrderByDescending(pair => pair.Sim)
                .Take(k)
                .Select(pair => pair.Doc)
                .ToArray();
            return matchedDocuments;
            ;
        }

        private double Tf(double termFrequency, bool sublinear)
        {
            return sublinear ? Math.Log(termFrequency) + 1 : termFrequency;
        }

        private double Idf(int token, bool smooth)
        {
            var ifreq = smooth
                ? 1.0 * (1 + DocumentCount) / (1 + DocumentFrequency[token])
                : 1.0 * DocumentCount / DocumentFrequency[token];
            return 1 + Math.Log(ifreq);
        }

        /// <summary>
        /// TF-IDF value of one token given the term frequency.
        /// </summary>
        private double TfIdf(double termFrequency, int token, bool sublinearTf, bool smoothIdf, bool noIdf)
        {
            return noIdf
                ? Tf(termFrequency, sublinearTf)
                : Tf(termFrequency, sublinearTf) * Idf(token, smoothIdf);
        }

        /// <summary>
        /// TF-IDF value of all tokens in a document.
        /// </summary>
        /// <returns>The tf-idf vector over the whole vocabulary.</returns>
        private double[] TfIdf(IList<int> documentTokens, bool sublinearTf = false, bool smoothIdf = true, bool noIdf = false,
            bool maxNormTf = false)
        {
            // Remove <CLS> <SEP> <PAD> <UNK>
            documentTokens = documentTokens.SkipWhile(token => token < 4).ToArray();

            // Compute term frequency
            var frequency = new int[VocabularySize];
            foreach (var token in documentTokens)
            {
                ++frequency[token];
            }

            var weight = new double[VocabularySize];
            var maxFrequency = frequency.Max();
            foreach (var token in documentTokens.ToHashSet())
            {
                var freq = maxNormTf ? MaxNormSmoothing + (1 - MaxNormSmoothing) * frequency[token] / maxFrequency : frequency[token];
                weight[token] = TfIdf(freq, token, sublinearTf, smoothIdf, noIdf);
            }
            return weight;
        }

        private double CosineSimilarityOfQuestion(IList<double> question, IList<double> document)
        {
            if (question.Count != document.Count)
            {
                throw new ArgumentException(
                    $"Vectors to compute cosine similarity must have the same length, got {question.Count} and {document.Count}");
            }

            var qMax = question.Max();
            var dMax = document.Max();
            var nominator = 0.0;
            var qSum = 0.0;
            var dSum = 0.0;
            for (var i = 0; i < question.Count; ++i)
            {
                var q = question[i] / qMax;
                var d = document[i] / dMax;
                if (!StopWordsTokenIds.Contains(i))
                {
                    nominator += q * d;
                    qSum += q * q;
                    dSum += d * d;
                }
            }
            return nominator / Math.Sqrt(qSum * dSum);
        }
    }
}