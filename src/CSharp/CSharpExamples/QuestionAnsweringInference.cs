// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Examples.Utils;
using TorchSharp;
using TorchSharp.Examples;

namespace CSharpExamples
{
    public class QuestionAnsweringInference
    {
        internal static void Run()
        {
            // Configure the run
            var config = new QuestionAnsweringConfig
            {
                LoadModelPath = "roberta-bertformat-model_weights.dat",
                DataDir = "data",
                TrainFile = "mixed_train.json",
                ValidFile = "mixed_valid.json",
                TestFile = "test.json",
                VocabDir = "vocab_files",

                BatchSize = 8,
                OptimizeSteps = 1,
                MaxSequence = 384,
                Cuda = true,
                SaveDir = "saved_models",

                LearningRate = 3e-5,
                LogEveryNSteps = 10,
                ValidateEveryNSteps = 2000,
                TopK = 5
            };
            Directory.CreateDirectory(config.SaveDir);

            // Initialize Model, Optimizer and Data Pre-processors
            var runner = new QuestionAnsweringInference(config);

            // Load Pre-trained General Purpose Model
            runner.LoadModel(config.LoadModelPath);

            // Load Corpus from Disk
            var corpus = runner.LoadCorpus(Path.Join(config.DataDir, config.TestFile));

            // Start Inference Loop
            runner.SearchOverCorpus(corpus);
        }

        private static readonly Logger<QuestionAnsweringInference> _logger = new();
        private const string _exit = "exit";

        private QuestionAnsweringConfig Config { get; }
        private RobertaForQuestionAnswering Model { get; }
        private RobertaTokenizer Tokenizer { get; }
        private RobertaInputBuilder InputBuilder { get; }

        private QuestionAnsweringInference(QuestionAnsweringConfig config)
        {
            Config = config;

            Model = new RobertaForQuestionAnswering(
                numLayers: 12,
                numAttentionHeads: 12,
                numEmbeddings: 50265,
                embeddingSize: 768,
                hiddenSize: 768,
                outputSize: 768,
                ffnHiddenSize: 3072,
                maxPositions: 512,
                maxTokenTypes: 2,
                layerNormEps: 1e-12,
                embeddingDropoutRate: 0.1,
                attentionDropoutRate: 0.1,
                attentionOutputDropoutRate: 0.1,
                outputDropoutRate: 0.1);
            if (config.Cuda) Model.cuda();

            Tokenizer = new RobertaTokenizer(config.VocabDir);
            InputBuilder = new RobertaInputBuilder(Tokenizer, config.MaxSequence);
        }

        public void LoadModel(string path)
        {
            _logger.Log($"Loading model from {path}...", newline: false);
            Model.load(path, false);
            if (Config.Cuda) Model.cuda();
            _logger.LogAppend("Done.");
        }

        public SquadCorpus LoadCorpus(string path)
        {
            return new SquadCorpus(path, Tokenizer, InputBuilder);
        }

        private void ModelForward(SquadSampleBatch batch, bool applyPredictMasks,
            out int trueBatchSize, out torch.Tensor startLogits, out torch.Tensor endLogits,
            out torch.Tensor startPositions, out torch.Tensor endPositions)
        {
            trueBatchSize = (int)batch.Tokens.size(0);
            (startLogits, endLogits) = Model.forward(batch.Tokens, batch.Positions, batch.Segments, batch.AttentionMasks);
            if (applyPredictMasks)
            {
                startLogits = startLogits.add_(batch.PredictMasks);
                endLogits = endLogits.add_(batch.PredictMasks);
            }

            startPositions = null;
            endPositions = null;
            if (batch.Starts.IsNotNull())
            {
                var ignoreIndex = startLogits.size(-1);
                startPositions = batch.Starts.view(-1).clamp(0, ignoreIndex);
                endPositions = batch.Ends.view(-1).clamp(0, ignoreIndex);
            }
        }

        /// <summary>
        /// Save GPU memory usage following this passage:
        /// https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#pre-allocate-memory-in-case-of-variable-input-length
        /// </summary>
        private void GpuMemoryWarmupOnlyForward()
        {
            using var disposeScope = torch.NewDisposeScope();
            Model.eval();
            var batch = SquadCorpus.GetMaxDummyBatch(Config);
            ModelForward(batch, false, out var trueBatchSize, out var startLogits, out var endLogits,
                out var startPositions, out var endPositions);
        }

        public void SearchOverCorpus(SquadCorpus corpus)
        {
            var serializerOptions = new JsonSerializerOptions
            {
                WriteIndented = true
            };
            var selector = new TfIdfDocumentSelector(corpus.Documents, Tokenizer);

            using var _ = torch.no_grad();
            GpuMemoryWarmupOnlyForward();
            Model.eval();
            while (true)
            {
                Console.Clear();
                Console.Write($"\nType your question (\"{_exit}\" to exit): ");
                var question = Console.ReadLine();
                if (question == _exit) break;

                var questionTokenIds = Tokenizer.TokenizeToId(question);
                var questionLength = questionTokenIds.Count + 2;

                var answers = new List<PredictionAnswer>();
                var bestMatch = selector.TopK(question, Config.TopK);
                foreach (var batch in corpus.GetBatches(Config, questionTokenIds, bestMatch.Take(1).ToArray()))
                {
                    using var disposeScope = torch.NewDisposeScope();
                    ModelForward(batch, true, out var trueBatchSize, out var startLogits, out var endLogits,
                            out var startPositions, out var endPositions);
                    for (var i = 0; i < trueBatchSize; ++i)
                    {
                        var (predictStartScores, predictStarts) = startLogits[i].topk(Config.TopK);
                        var (predictEndScores, predictEnds) = endLogits[i].topk(Config.TopK);
                        var topKSpans = SquadMetric.ComputeTopKSpansWithScore(predictStartScores, predictStarts, predictEndScores, predictEnds, Config.TopK);
                        var predictStart = topKSpans[0].start;
                        var predictEnd = topKSpans[0].end;

                        // Restore predicted answer text
                        var document = bestMatch[i];
                        var contextText = Tokenizer.Untokenize(document.ContextTokens);
                        foreach (var (start, end, score) in topKSpans)
                        {
                            var answerText = Tokenizer.Untokenize(
                                document.ContextTokens.ToArray()[(start - questionLength)..(end - questionLength + 1)]);
                            answers.Add(new PredictionAnswer { Score = score, Text = answerText });
                        }
                    }

                    answers = answers.OrderByDescending(answer => answer.Score).Take(Config.TopK).ToList();
                    var outputString = JsonSerializer.Serialize(answers, serializerOptions);
                    Console.WriteLine($"Predictions:\n{outputString}");
                } // end foreach
            } // end while
        }
    }

    internal struct PredictionAnswer
    {
        public string Text { get; set; }
        public double Score { get; set; }
    }
}