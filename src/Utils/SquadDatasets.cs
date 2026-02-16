// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using TorchSharp;

namespace Examples.Utils
{
    /// <summary>
    /// Store the extracted samples of the SQuAD v2.0 data set.
    /// The training and development data can be downloaded from the page at: https://rajpurkar.github.io/SQuAD-explorer/
    /// </summary>
    public class SquadDataset
    {
        private static readonly Logger<SquadDataset> _logger = new();

        private readonly int[] ZeroPad;
        private readonly int[] NegBillionPad;
        private readonly int[] TokenPad;

        private readonly RobertaTokenizer robertaTokenizer;
        private readonly RobertaInputBuilder RobertaInputBuilder;
        public string FilePath { get; }
        public List<SquadSample> Samples { get; }

        public int Count => Samples.Count;

        public SquadDataset(string filePath, RobertaTokenizer robertaTokenizer, RobertaInputBuilder inputBuilder)
        {
            const int negBillion = (int)-1e9;
            ZeroPad = Enumerable.Repeat(0, inputBuilder.MaxPositions).ToArray();
            NegBillionPad = Enumerable.Repeat(negBillion, inputBuilder.MaxPositions).ToArray();
            TokenPad = Enumerable.Repeat(robertaTokenizer.PadIndex, inputBuilder.MaxPositions).ToArray();

            FilePath = filePath;
            this.robertaTokenizer = robertaTokenizer;
            RobertaInputBuilder = inputBuilder;
            Samples = LoadDataset(filePath);
        }

        private List<SquadSample> LoadDataset(string filePath)
        {
            var samples = new List<SquadSample>();
            var dataset = JsonSerializer.Deserialize<SquadDataFromFile>(File.ReadAllText(filePath));
            var start = DateTime.Now;
            var count = 1;
            foreach (var document in dataset.data)
            {
                var end = DateTime.Now;
                _logger.LogLoop($"Processing document {count}/{dataset.data.Count}, time consumed {end - start}");

                foreach (var paragraph in document.paragraphs)
                {
                    var contextTokens = robertaTokenizer.Tokenize(paragraph.context);
                    var contextTokenIds = robertaTokenizer.TokensToIds(contextTokens);
                    var mapping = AlignAnswerPosition(contextTokens, paragraph.context);
                    if (mapping == null)
                    {
                        continue;
                    }

                    foreach (var qapair in paragraph.qas)
                    {
                        var questionTokenIds = robertaTokenizer.TokenizeToId(qapair.question);

                        // Treat plausible answers as normal answers, which is not a normal behavior.
                        //var answerList = qapair.is_impossible ? qapair.plausible_answers : qapair.answers;
                        //var answerPositions = new List<(int, int)>();
                        //foreach (var answer in answerList)
                        //{
                        //    // inclusive [start, end]
                        //    var startPosition = mapping[answer.answer_start];
                        //    var endPosition = mapping[answer.answer_start + answer.text.Length - 1];
                        //    answerPositions.Add((startPosition, endPosition));
                        //}

                        // Treat plausible answers as "no answers".
                        var answerPositions = new List<(int, int)>();
                        if (!qapair.is_impossible)
                        {
                            foreach (var answer in qapair.answers)
                            {
                                // inclusive [start, end]
                                var startPosition = mapping[answer.answer_start];
                                var endPosition = mapping[answer.answer_start + answer.text.Length - 1];
                                answerPositions.Add((startPosition, endPosition));
                            }
                        }

                        samples.Add(new SquadSample(
                            title: document.title,
                            url: document.url,
                            context: paragraph.context,
                            contextTokens: contextTokenIds,
                            id: qapair.id,
                            question: qapair.question,
                            questionTokens: questionTokenIds,
                            answers: answerPositions));
                    }
                }

                count++;
            }
            _logger.LogAppend();

            return samples;
        }

        private Dictionary<int, int> AlignAnswerPosition(IList<string> tokens, string text)
        {
            var mapping = new Dictionary<int, int>();
            int surrogateDeduce = 0;
            for (var (i, j, tid) = (0, 0, 0); i < text.Length && tid < tokens.Count;)
            {
                // Move to a new token
                if (j >= tokens[tid].Length)
                {
                    ++tid;
                    j = 0;
                }
                // There are a few UTF-32 chars in corpus, which is considered one char in position
                else if (i + 1 < text.Length && char.IsSurrogatePair(text[i], text[i + 1]))
                {
                    i += 2;
                    ++surrogateDeduce;
                }
                // White spaces are not included in tokens
                else if (char.IsWhiteSpace(text[i]))
                {
                    ++i;
                }
                // Chars not included in tokenizer will not appear in tokens
                else if (!robertaTokenizer.ByteToUnicode.ContainsKey(text[i]))
                {
                    mapping[i - surrogateDeduce] = tid;
                    ++i;
                }
                // "\\\"", "``" and "''" converted to "\"" in normalizer
                else if (i + 1 < text.Length && tokens[tid][j] == '"'
                    && ((text[i] == '`' && text[i + 1] == '`')
                     || (text[i] == '\'' && text[i + 1] == '\'')
                     || (text[i] == '\\' && text[i + 1] == '"')))
                {
                    mapping[i - surrogateDeduce] = mapping[i + 1 - surrogateDeduce] = tid;
                    i += 2;
                    j += 1;
                }
                // Normal match
                else if (text[i] == tokens[tid][j])
                {
                    mapping[i - surrogateDeduce] = tid;
                    ++i;
                    ++j;
                }
                // There are a few real \u0120 chars in the corpus, so this rule has to be later than text[i] == tokens[tid][j].
                else if (tokens[tid][j] == '\u0120' && j == 0)
                {
                    ++j;
                }
                else
                {
                    throw new Exception("unmatched!");
                }
            }

            return mapping;
        }

        public IEnumerable<SquadSampleBatch> GetBatches(QuestionAnsweringConfig config, bool shuffle)
        {
            var buffer = new List<(int[], int[], int[], int[], int[], int)>();
            var samples = Samples;
            if (shuffle)
            {
                var random = new Random();
                samples = samples.OrderBy(_ => random.Next()).ToList();
            }
            foreach (var sample in samples)
            {
                (int[] tokens, int[] positions, int[] segments, int[] startPositions, int[] endPositions, int questionSegmentLength) ioArrays
                    = RobertaInputBuilder.Build(sample.QuestionTokens, sample.ContextTokens, sample.Answers);

                buffer.Add(ioArrays);
                if (buffer.Count == config.BatchSize)
                {
                    yield return BufferToBatch(buffer, config.Cuda);
                    buffer.Clear();
                }
            }
            if (buffer.Count > 0)
            {
                yield return BufferToBatch(buffer, config.Cuda);
                buffer.Clear();
            }
        }

        public static SquadSampleBatch GetMaxDummyBatch(QuestionAnsweringConfig config)
        {
            var batchSize = config.BatchSize;
            var maxLength = config.MaxSequence;
            var maxAnswer = 1;
            var device = config.Cuda ? torch.CUDA : torch.CPU;

            return new SquadSampleBatch
            {
                Tokens = torch.zeros(batchSize, maxLength, dtype: torch.int64, device: device),
                Positions = torch.zeros(batchSize, maxLength, dtype: torch.int64, device: device),
                Segments = torch.zeros(batchSize, maxLength, dtype: torch.int64, device: device),
                Starts = torch.zeros(batchSize, maxAnswer, dtype: torch.int64, device: device),
                Ends = torch.zeros(batchSize, maxAnswer, dtype: torch.int64, device: device),
                AttentionMasks = torch.zeros(batchSize, 1, 1, maxLength, dtype: torch.float32, device: device),
                PredictMasks = null,
            };
        }

        private SquadSampleBatch BufferToBatch(List<(int[], int[], int[], int[], int[], int)> buffer, bool cuda)
        {
            using var disposeScope = torch.NewDisposeScope();
            var device = cuda ? torch.CUDA : torch.CPU;
            var maxLength = buffer.Max(tensors => tensors.Item1.Length);
            var maxAnswer = buffer.Max(tensors => tensors.Item4.Length);
            var tokens = new torch.Tensor[buffer.Count];
            var positions = new torch.Tensor[buffer.Count];
            var segments = new torch.Tensor[buffer.Count];
            var starts = new torch.Tensor[buffer.Count];
            var ends = new torch.Tensor[buffer.Count];
            var attentionMasks = new torch.Tensor[buffer.Count];
            var predictMasks = new torch.Tensor[buffer.Count];
            for (var i = 0; i < buffer.Count; ++i)
            {
                var arrays = buffer[i];
                var length = arrays.Item1.Length;
                var questionSegmentLength = arrays.Item6;
                var token = torch.tensor(arrays.Item1.Concat(TokenPad[..(maxLength - length)]).ToArray(),
                    1, maxLength, dtype: torch.int64, device: device);
                var position = torch.tensor(arrays.Item2.Concat(ZeroPad[..(maxLength - length)]).ToArray(),
                    1, maxLength, dtype: torch.int64, device: device);
                var segment = torch.tensor(arrays.Item3.Concat(ZeroPad[..(maxLength - length)]).ToArray(),
                    1, maxLength, dtype: torch.int64, device: device);
                var attentionMask = torch.tensor(ZeroPad[..length].Concat(NegBillionPad[..(maxLength - length)]).ToArray(),
                    new long[] { 1, 1, 1, maxLength }, dtype: torch.float32, device: device);
                var predictMask = torch.tensor(
                    NegBillionPad[..questionSegmentLength].Concat(ZeroPad[..(length - questionSegmentLength - 1)])
                        .Concat(NegBillionPad[..(maxLength - length + 1)]).ToArray(),
                    1, maxLength, dtype: torch.float32, device: device);

                var answer = arrays.Item4.Length;
                var start = torch.tensor(arrays.Item4.Concat(Enumerable.Repeat(arrays.Item4[^1], maxAnswer - answer)).ToArray(),
                    1, maxAnswer, dtype: torch.int64, device: device);
                var end = torch.tensor(arrays.Item5.Concat(Enumerable.Repeat(arrays.Item5[^1], maxAnswer - answer)).ToArray(),
                    1, maxAnswer, dtype: torch.int64, device: device);

                tokens[i] = token;
                positions[i] = position;
                segments[i] = segment;
                starts[i] = start;
                ends[i] = end;
                attentionMasks[i] = attentionMask;
                predictMasks[i] = predictMask;
            }

            return new SquadSampleBatch
            {
                Tokens = torch.cat(tokens, dimension: 0).MoveToOuterDisposeScope(),
                Positions = torch.cat(positions, dimension: 0).MoveToOuterDisposeScope(),
                Segments = torch.cat(segments, dimension: 0).MoveToOuterDisposeScope(),
                Starts = torch.cat(starts, dimension: 0).MoveToOuterDisposeScope(),
                Ends = torch.cat(ends, dimension: 0).MoveToOuterDisposeScope(),
                AttentionMasks = torch.cat(attentionMasks, dimension: 0).MoveToOuterDisposeScope(),
                PredictMasks = torch.cat(predictMasks, dimension: 0).MoveToOuterDisposeScope(),
            };
        }
    }

    /// <summary>
    /// Store the extracted document corpus from a data set file with the same format as SQuAD v2.0.
    /// The SQuAD v2.0 data can be downloaded from the page at: https://rajpurkar.github.io/SQuAD-explorer/
    /// </summary>
    public class SquadCorpus
    {
        private static readonly Logger<SquadCorpus> _logger = new();

        private readonly int[] ZeroPad;
        private readonly int[] NegBillionPad;
        private readonly int[] TokenPad;

        private readonly RobertaTokenizer robertaTokenizer;
        private readonly RobertaInputBuilder RobertaInputBuilder;
        public readonly string FilePath;
        public readonly IReadOnlyList<SquadDocument> Documents;

        public int Count => Documents.Count;

        public SquadCorpus(string filePath, RobertaTokenizer robertaTokenizer, RobertaInputBuilder inputBuilder)
        {
            int negBillion = (int)-1e9;
            ZeroPad = Enumerable.Repeat(0, inputBuilder.MaxPositions).ToArray();
            NegBillionPad = Enumerable.Repeat(negBillion, inputBuilder.MaxPositions).ToArray();
            TokenPad = Enumerable.Repeat(robertaTokenizer.PadIndex, inputBuilder.MaxPositions).ToArray();

            FilePath = filePath;
            this.robertaTokenizer = robertaTokenizer;
            RobertaInputBuilder = inputBuilder;
            Documents = LoadDocuments(filePath);
        }

        private List<SquadDocument> LoadDocuments(string filePath)
        {
            var documents = new List<SquadDocument>();
            var dataset = JsonSerializer.Deserialize<SquadDataFromFile>(File.ReadAllText(filePath));
            var start = DateTime.Now;
            var count = 1;
            foreach (var document in dataset.data)
            {
                var end = DateTime.Now;
                _logger.LogLoop($"Processing document {count}/{dataset.data.Count}, time consumed {end - start}");

                foreach (var paragraph in document.paragraphs)
                {
                    var contextTokenIds = robertaTokenizer.TokenizeToId(paragraph.context, @"\s+");

                    documents.Add(new SquadDocument(
                        title: document.title,
                        url: document.url,
                        context: paragraph.context,
                        contextTokens: contextTokenIds));
                }

                count++;
            }
            _logger.LogAppend();

            return documents;
        }

        public static SquadSampleBatch GetMaxDummyBatch(QuestionAnsweringConfig config)
        {
            using var disposeScope = torch.NewDisposeScope();
            var batchSize = config.BatchSize;
            var maxLength = config.MaxSequence;
            var device = config.Cuda ? torch.CUDA : torch.CPU;

            var token = torch.zeros(batchSize, maxLength, dtype: torch.int64, device: device);
            var position = torch.zeros(batchSize, maxLength, dtype: torch.int64, device: device);
            var segment = torch.zeros(batchSize, maxLength, dtype: torch.int64, device: device);
            var attentionMask = torch.zeros(batchSize, 1, 1, maxLength, dtype: torch.float32, device: device);

            return new SquadSampleBatch
            {
                Tokens = token.MoveToOuterDisposeScope(),
                Positions = position.MoveToOuterDisposeScope(),
                Segments = segment.MoveToOuterDisposeScope(),
                Starts = null,
                Ends = null,
                AttentionMasks = attentionMask.MoveToOuterDisposeScope(),
                PredictMasks = null
            };
        }

        public IEnumerable<SquadSampleBatch> GetBatches(QuestionAnsweringConfig config, string questionText)
        {
            // parse question
            var questionTokenIds = robertaTokenizer.TokenizeToId(questionText);

            foreach (var batch in GetBatches(config, questionTokenIds, Documents))
            {
                yield return batch;
            }
        }

        public IEnumerable<SquadSampleBatch> GetBatches(QuestionAnsweringConfig config, IList<int> questionTokens, IReadOnlyList<SquadDocument> documents)
        {
            var buffer = new List<(int[], int[], int[], int)>();
            foreach (var document in documents)
            {
                var ioArrays = RobertaInputBuilder.Build(questionTokens, document.ContextTokens);
                buffer.Add(ioArrays);
                if (buffer.Count == config.BatchSize)
                {
                    yield return BufferToBatch(buffer, config.Cuda);
                    buffer.Clear();
                }
            }
            if (buffer.Count > 0)
            {
                yield return BufferToBatch(buffer, config.Cuda);
                buffer.Clear();
            }
        }

        private SquadSampleBatch BufferToBatch(List<(int[], int[], int[], int)> buffer, bool cuda)
        {
            using var disposeScope = torch.NewDisposeScope();
            var device = cuda ? torch.CUDA : torch.CPU;
            var maxLength = buffer.Max(tensors => tensors.Item1.Length);
            var tokens = new torch.Tensor[buffer.Count];
            var positions = new torch.Tensor[buffer.Count];
            var segments = new torch.Tensor[buffer.Count];
            var attentionMasks = new torch.Tensor[buffer.Count];
            var predictMasks = new torch.Tensor[buffer.Count];
            for (var i = 0; i < buffer.Count; ++i)
            {
                var arrays = buffer[i];
                var length = arrays.Item1.Length;
                var questionSegmentLength = arrays.Item4;
                var token = torch.tensor(arrays.Item1.Concat(TokenPad[..(maxLength - length)]).ToArray(),
                    1, maxLength, dtype: torch.int64, device: device);
                var position = torch.tensor(arrays.Item2.Concat(ZeroPad[..(maxLength - length)]).ToArray(),
                    1, maxLength, dtype: torch.int64, device: device);
                var segment = torch.tensor(arrays.Item3.Concat(ZeroPad[..(maxLength - length)]).ToArray(),
                    1, maxLength, dtype: torch.int64, device: device);
                var attentionMask = torch.tensor(ZeroPad[..length].Concat(NegBillionPad[..(maxLength - length)]).ToArray(),
                    new long[] { 1, 1, 1, maxLength }, dtype: torch.float32, device: device);
                var predictMask = torch.tensor(
                    NegBillionPad[..questionSegmentLength].Concat(ZeroPad[..(length - questionSegmentLength - 1)])
                        .Concat(NegBillionPad[..(maxLength - length + 1)]).ToArray(),
                    1, maxLength, dtype: torch.float32, device: device);

                tokens[i] = token;
                positions[i] = position;
                segments[i] = segment;
                attentionMasks[i] = attentionMask;
                predictMasks[i] = predictMask;
            }

            return new SquadSampleBatch
            {
                Tokens = torch.cat(tokens, dimension: 0).MoveToOuterDisposeScope(),
                Positions = torch.cat(positions, dimension: 0).MoveToOuterDisposeScope(),
                Segments = torch.cat(segments, dimension: 0).MoveToOuterDisposeScope(),
                Starts = null,
                Ends = null,
                AttentionMasks = torch.cat(attentionMasks, dimension: 0).MoveToOuterDisposeScope(),
                PredictMasks = torch.cat(predictMasks, dimension: 0).MoveToOuterDisposeScope(),
            };
        }
    }

    public class SquadDocument
    {
        public string Title { get; }
        public string Url { get; }
        public string Context { get; }
        public IList<int> ContextTokens { get; }

        public SquadDocument(string title, string url, string context, IList<int> contextTokens)
        {
            Title = title;
            Url = url;
            Context = context;
            ContextTokens = contextTokens;
        }
    }

    public class SquadSample
    {
        public string Title { get; }
        public string Url { get; }
        public string Context { get; }
        public IList<int> ContextTokens { get; }
        public string Id { get; }
        public string Question { get; }
        public IList<int> QuestionTokens { get; }
        public IList<(int, int)> Answers { get; }

        public SquadSample(string title, string url, string context, IList<int> contextTokens, string id,
            string question, IList<int> questionTokens, IList<(int, int)> answers)
        {
            Title = title;
            Url = url;
            Context = context;
            ContextTokens = contextTokens;
            Id = id;
            Question = question;
            QuestionTokens = questionTokens;
            Answers = answers;
        }
    }

    public struct SquadSampleBatch
    {
        public torch.Tensor Tokens;
        public torch.Tensor Positions;
        public torch.Tensor Segments;
        public torch.Tensor Starts;
        public torch.Tensor Ends;
        public torch.Tensor AttentionMasks;
        public torch.Tensor PredictMasks;
    }

    internal struct SquadDataFromFile
    {
        public string version { get; set; }
        public IList<SquadData> data { get; set; }
    }

    internal struct SquadData
    {
        public string title { get; set; }
        public string url { get; set; }
        public IList<SquadParagraph> paragraphs { get; set; }
    }

    internal struct SquadParagraph
    {
        public string context { get; set; }
        public IList<SquadQAPair> qas { get; set; }
    }

    internal struct SquadQAPair
    {
        public string question { get; set; }
        public string id { get; set; }
        public IList<SquadAnswer> answers { get; set; }
        public IList<SquadAnswer> plausible_answers { get; set; }
        public bool is_impossible { get; set; }
    }

    internal struct SquadAnswer
    {
        public string text { get; set; }
        public int answer_start { get; set; }
    }
}
