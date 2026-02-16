// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Diagnostics;

using TorchSharp;
using TorchSharp.Examples;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace CSharpExamples
{
    /// <summary>
    /// Time Sequence Prediction using LSTM
    ///
    /// Based on: https://github.com/pytorch/examples/tree/main/time_sequence_prediction
    ///
    /// Generates sine wave data with random phase shifts, trains a stacked LSTMCell model
    /// to predict the next value, and then predicts future values beyond the training data.
    /// Uses synthetic data — no dataset download needed.
    /// </summary>
    public class TimeSequencePrediction
    {
        private const int T = 20;
        private const int L = 1000;
        private const int N = 100;

        internal static void Run(int epochs, int timeout, string logdir)
        {
            var device =
                torch.cuda.is_available() ? torch.CUDA :
                torch.mps_is_available() ? torch.MPS :
                torch.CPU;

            Console.WriteLine();
            Console.WriteLine($"\tRunning TimeSequencePrediction on {device.type.ToString()} for {epochs} epochs, terminating after {TimeSpan.FromSeconds(timeout)}.");
            Console.WriteLine();

            torch.random.manual_seed(0);

            // Generate sine wave training data (matching PyTorch's generate_sine_wave.py)
            Console.WriteLine($"\tGenerating sine wave training data...");
            var data = GenerateSineWaveData();

            var input = data[TensorIndex.Slice(3, null), TensorIndex.Slice(null, -1)];
            var target = data[TensorIndex.Slice(3, null), TensorIndex.Slice(1, null)];
            var test_input = data[TensorIndex.Slice(null, 3), TensorIndex.Slice(null, -1)];
            var test_target = data[TensorIndex.Slice(null, 3), TensorIndex.Slice(1, null)];

            // Move to device
            input = input.to(device);
            target = target.to(device);
            test_input = test_input.to(device);
            test_target = test_target.to(device);

            Console.WriteLine($"\tCreating the model...");
            Console.WriteLine();

            var model = new SequenceModel("time-seq", device);
            model.to(torch.float64);

            var criterion = MSELoss();
            var optimizer = torch.optim.LBFGS(model.parameters(), lr: 0.8);

            var writer = String.IsNullOrEmpty(logdir) ? null : torch.utils.tensorboard.SummaryWriter(logdir, createRunName: true);

            Stopwatch totalTime = new Stopwatch();
            totalTime.Start();

            for (var epoch = 0; epoch < epochs; epoch++)
            {
                using (var d = torch.NewDisposeScope())
                {
                    Console.WriteLine($"STEP: {epoch}");

                    // Training step with LBFGS closure
                    Tensor lastLoss = null;

                    Tensor closure()
                    {
                        optimizer.zero_grad();
                        var output = model.forward(input, 0);
                        var loss = criterion.forward(output, target);
                        Console.WriteLine($"\tloss: {loss.item<double>():F6}");
                        loss.backward();
                        lastLoss = loss;
                        return loss;
                    }

                    optimizer.step(closure);

                    // Test: predict with future steps
                    using (torch.no_grad())
                    {
                        var future = 1000;
                        var pred = model.forward(test_input, future);
                        var loss = criterion.forward(pred[TensorIndex.Colon, TensorIndex.Slice(null, -future)], test_target);
                        Console.WriteLine($"\ttest loss: {loss.item<double>():F6}");

                        if (writer != null)
                        {
                            writer.add_scalar("time_seq/train_loss", (float)lastLoss.item<double>(), epoch);
                            writer.add_scalar("time_seq/test_loss", (float)loss.item<double>(), epoch);
                        }
                    }

                    if (totalTime.Elapsed.TotalSeconds > timeout) break;
                }
            }

            totalTime.Stop();
            Console.WriteLine($"\nElapsed time: {totalTime.Elapsed.TotalSeconds:F1} s.");
        }

        /// <summary>
        /// Generates sine wave data matching PyTorch's generate_sine_wave.py.
        /// Creates N sine waves of length L with random phase offsets.
        /// </summary>
        private static Tensor GenerateSineWaveData()
        {
            var rng = new Random(2);
            var x = new double[N, L];

            for (int i = 0; i < N; i++)
            {
                var offset = rng.Next(-4 * T, 4 * T);
                for (int j = 0; j < L; j++)
                {
                    x[i, j] = Math.Sin((j + offset) / (double)T);
                }
            }

            return torch.tensor(x, dtype: torch.float64);
        }
    }
}
