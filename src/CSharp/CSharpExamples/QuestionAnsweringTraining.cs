// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.IO;
using System.Linq;
using Examples.Utils;
using TorchSharp;
using TorchSharp.Examples;
using TorchSharp.Modules;

namespace CSharpExamples
{
    public class QuestionAnsweringTraining
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
            var runner = new QuestionAnsweringTraining(config);

            // Load Pre-trained General Purpose Model
            runner.LoadModel(config.LoadModelPath);

            // Load Dataset from Disk
            var trainDataset = runner.LoadDataset(Path.Join(config.DataDir, config.TrainFile));
            var validDataset = runner.LoadDataset(Path.Join(config.DataDir, config.ValidFile));
            var testDataset = runner.LoadDataset(Path.Join(config.DataDir, config.TestFile));

            // Start Training (Finetuning)
            runner.Train(trainDataset, validDataset, testDataset);
        }

        private static readonly Logger<QuestionAnsweringTraining> _logger = new();

        private QuestionAnsweringConfig Config { get; }
        private RobertaForQuestionAnswering Model { get; }
        private RobertaTokenizer Tokenizer { get; }
        private RobertaInputBuilder InputBuilder { get; }

        private AdamW Optimizer { get; }

        private QuestionAnsweringTraining(QuestionAnsweringConfig config)
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
            Optimizer = torch.optim.AdamW(Model.parameters(), Config.LearningRate);

            Tokenizer = new RobertaTokenizer(config.VocabDir);
            InputBuilder = new RobertaInputBuilder(Tokenizer, config.MaxSequence);
        }

        private void LoadModel(string path)
        {
            _logger.Log($"Loading model from {path}...", newline: false);
            Model.load(path, false);
            if (Config.Cuda) Model.cuda();
            _logger.LogAppend("Done.");
        }

        private SquadDataset LoadDataset(string path)
        {
            return new SquadDataset(path, Tokenizer, InputBuilder);
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
        private void GpuMemoryWarmup()
        {
            using var disposeScope = torch.NewDisposeScope();

            Model.train();
            var batch = SquadDataset.GetMaxDummyBatch(Config);
            ModelForward(batch, false, out var trueBatchSize, out var startLogits, out var endLogits,
                out var startPositions, out var endPositions);
            var lossFunc = torch.nn.functional.cross_entropy_loss();
            var lossStart = lossFunc(startLogits, startPositions);
            var lossEnd = lossFunc(endLogits, endPositions);
            var loss = ((lossStart + lossEnd) / 2);
            loss.backward();
            Optimizer.zero_grad();
        }

        public void Train(SquadDataset trainDataset, SquadDataset validDataset, SquadDataset testDataset)
        {
            var batchSize = Config.BatchSize;
            var optimizeSteps = Config.OptimizeSteps;

            var correctSamples = 0;
            var startCorrectSamples = 0;
            var endCorrectSamples = 0;
            var totalSamples = 0;
            var all = (int)Math.Ceiling(1.0 * trainDataset.Count / batchSize);
            var totalLoss = 0.0;

            var step = 1;
            var startTime = DateTime.Now;
            GpuMemoryWarmup();
            for (var epoch = 0; epoch < 10; epoch++)
            {
                foreach (var batch in trainDataset.GetBatches(Config, shuffle: true))
                {
                    using var disposeScope = torch.NewDisposeScope();
                    Model.train();
                    Optimizer.zero_grad();

                    ModelForward(batch, false, out var trueBatchSize, out var startLogits, out var endLogits,
                        out var startPositions, out var endPositions);

                    var lossFunc = torch.nn.functional.cross_entropy_loss();
                    var lossStart = lossFunc(startLogits, startPositions);
                    var lossEnd = lossFunc(endLogits, endPositions);
                    var loss = ((lossStart + lossEnd) / 2);
                    // For N-step optimization
                    totalLoss += loss.ToItem<float>();
                    loss /= optimizeSteps;
                    loss.backward();
                    if (step % optimizeSteps == 0)
                    {
                        var temp = Optimizer.parameters().ToArray().Select(p => p.grad()).ToArray();
                        Optimizer.step();
                    }

                    var predictionStarts = startLogits.argmax(-1).view(-1).ToArray<long>();
                    var predictionEnds = endLogits.argmax(-1).view(-1).ToArray<long>();
                    var groundStarts = startPositions.ToArray<long>();
                    var groundEnds = endPositions.ToArray<long>();
                    for (var i = 0; i < trueBatchSize; ++i)
                    {
                        if (predictionStarts[i] == groundStarts[i] && predictionEnds[i] == groundEnds[i])
                        {
                            ++correctSamples;
                            ++startCorrectSamples;
                            ++endCorrectSamples;
                        }
                        else if (predictionStarts[i] == groundStarts[i])
                        {
                            ++startCorrectSamples;
                        }
                        else if (predictionEnds[i] == groundEnds[i])
                        {
                            ++endCorrectSamples;
                        }
                    }
                    totalSamples += trueBatchSize;

                    if (step % Config.LogEveryNSteps == 0)
                    {
                        var endTime = DateTime.Now;
                        _logger.LogLoop($"                                                              " +
                            $"\rEpoch {epoch} - {step % all}/{all}," +
                            $"\t Time {endTime - startTime}," +
                            $"\t Acc {(totalSamples == 1 ? 0.0 : 1.0 * correctSamples / (totalSamples - 1)):N4}" +
                            $"\t StartAcc {(totalSamples == 1 ? 0.0 : 1.0 * startCorrectSamples / (totalSamples - 1)):N4}" +
                            $"\t EndAcc {(totalSamples == 1 ? 0.0 : 1.0 * endCorrectSamples / (totalSamples - 1)):N4}");
                    }

                    if (step % Config.ValidateEveryNSteps == 0)
                    {
                        _logger.LogAppend($"\t Loss {totalLoss / Config.ValidateEveryNSteps:N4}");
                        Validate(validDataset);
                        Validate(testDataset);
                        correctSamples = startCorrectSamples = endCorrectSamples = totalSamples = 0;
                        totalLoss = 0.0;
                        Model.save(Path.Join(Config.SaveDir, $"model_{step}.tsm"));
                    }
                    ++step;
                }
            }
        }

        public void Validate(SquadDataset validDataset)
        {
            _logger.Log($"Evaluating on {validDataset.FilePath}...", newline: false);

            Model.eval();
            var correct = 0;
            var startCorrect = 0;
            var endCorrect = 0;
            var f1score = 0.0;
            var total = 0;
            using (torch.no_grad())
            {
                foreach (var batch in validDataset.GetBatches(Config, shuffle: false))
                {
                    using var disposeScope = torch.NewDisposeScope();

                    ModelForward(batch, false, out var trueBatchSize, out var startLogits, out var endLogits,
                        out var startPositions, out var endPositions);

                    var predictionStarts = startLogits.argmax(-1).view(-1).ToArray<long>();
                    var predictionEnds = endLogits.argmax(-1).view(-1).ToArray<long>();
                    for (var i = 0; i < trueBatchSize; ++i)
                    {
                        var predictionStart = predictionStarts[i];
                        var predictionEnd = predictionEnds[i];
                        var zip = Enumerable.Zip(startPositions[i].ToArray<long>(), endPositions[i].ToArray<long>()).ToArray();
                        if (zip.Any(pair => predictionStart == pair.First && predictionEnd == pair.Second))
                        {
                            ++correct;
                            ++startCorrect;
                            ++endCorrect;
                        }
                        else if (zip.Any(pair => predictionStart == pair.First))
                        {
                            ++startCorrect;
                        }
                        else if (zip.Any(pair => predictionEnd == pair.Second))
                        {
                            ++endCorrect;
                        }
                        f1score += zip.Max(pair => SquadMetric.ComputeF1(predictionStart, predictionEnd, pair.First, pair.Second));
                    }
                    total += trueBatchSize;
                }
            }
            _logger.LogAppend(
                $"\rAccuracy: {1.0 * correct / total:N4}, {correct}/{total} ---- " +
                $"F1Score: {f1score / total:N4} ---- " +
                $"Start: {1.0 * startCorrect / total:N4}, {startCorrect} ---- " +
                $"End: {1.0 * endCorrect / total:N4}, {endCorrect}.");
        }
    }
}