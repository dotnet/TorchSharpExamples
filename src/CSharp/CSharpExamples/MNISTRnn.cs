// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.IO;
using System.Collections.Generic;
using System.Diagnostics;

using TorchSharp;
using static TorchSharp.torchvision;

using TorchSharp.Examples;
using TorchSharp.Examples.Utils;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace CSharpExamples
{
    /// <summary>
    /// MNIST classification using RNN (LSTM)
    ///
    /// Based on: https://github.com/pytorch/examples/tree/main/mnist_rnn
    ///
    /// Treats each MNIST image as a sequence of 28 rows, each with 28 features,
    /// and classifies using an LSTM network.
    /// </summary>
    /// <remarks>
    /// Uses the same MNIST dataset as the MNIST example.
    /// Download from: http://yann.lecun.com/exdb/mnist/
    /// </remarks>
    public class MNISTRnn
    {
        private static int _epochs = 14;
        private static int _trainBatchSize = 64;
        private static int _testBatchSize = 1000;

        private readonly static int _logInterval = 100;

        internal static void Run(int epochs, int timeout, string logdir)
        {
            _epochs = epochs;

            var device =
                torch.cuda.is_available() ? torch.CUDA :
                torch.mps_is_available() ? torch.MPS :
                torch.CPU;

            Console.WriteLine();
            Console.WriteLine($"\tRunning MNIST RNN on {device.type.ToString()} for {epochs} epochs, terminating after {TimeSpan.FromSeconds(timeout)}.");
            Console.WriteLine();

            torch.random.manual_seed(1);

            var dataset = "mnist";
            var datasetPath = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", dataset);

            var sourceDir = datasetPath;
            var targetDir = Path.Combine(datasetPath, "test_data");

            if (!Directory.Exists(targetDir))
            {
                Directory.CreateDirectory(targetDir);
                Decompress.DecompressGZipFile(Path.Combine(sourceDir, "train-images-idx3-ubyte.gz"), targetDir);
                Decompress.DecompressGZipFile(Path.Combine(sourceDir, "train-labels-idx1-ubyte.gz"), targetDir);
                Decompress.DecompressGZipFile(Path.Combine(sourceDir, "t10k-images-idx3-ubyte.gz"), targetDir);
                Decompress.DecompressGZipFile(Path.Combine(sourceDir, "t10k-labels-idx1-ubyte.gz"), targetDir);
            }

            if (device.type == DeviceType.CUDA)
            {
                _trainBatchSize *= 4;
                _testBatchSize *= 4;
            }

            Console.WriteLine($"\tCreating the model...");

            var model = new MNISTRNNModel("mnist-rnn", device);

            var normImage = transforms.Normalize(new double[] { 0.1307 }, new double[] { 0.3081 }, device: (Device)device);

            Console.WriteLine($"\tPreparing training and test data...");
            Console.WriteLine();

            using (MNISTReader train = new MNISTReader(targetDir, "train", _trainBatchSize, device: device, shuffle: true, transform: normImage),
                               test = new MNISTReader(targetDir, "t10k", _testBatchSize, device: device, transform: normImage))
            {
                var optimizer = torch.optim.Adadelta(model.parameters(), lr: 0.1);
                var scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.7);

                var writer = String.IsNullOrEmpty(logdir) ? null : torch.utils.tensorboard.SummaryWriter(logdir, createRunName: true);

                Stopwatch totalTime = new Stopwatch();
                totalTime.Start();

                for (var epoch = 1; epoch <= _epochs; epoch++)
                {
                    Train(model, optimizer, device, train, epoch, train.BatchSize, train.Size);
                    Test(model, writer, device, test, epoch, test.Size);

                    Console.WriteLine($"End-of-epoch memory use: {GC.GetTotalMemory(false)}");
                    scheduler.step();

                    if (totalTime.Elapsed.TotalSeconds > timeout) break;
                }

                totalTime.Stop();
                Console.WriteLine($"Elapsed time: {totalTime.Elapsed.TotalSeconds:F1} s.");
            }
        }

        private static void Train(
            Module<Tensor, Tensor> model,
            torch.optim.Optimizer optimizer,
            Device device,
            IEnumerable<(Tensor, Tensor)> dataLoader,
            int epoch,
            long batchSize,
            int size)
        {
            model.train();

            int batchId = 1;

            Console.WriteLine($"Epoch: {epoch}...");

            foreach (var (data, target) in dataLoader)
            {
                using (var d = torch.NewDisposeScope())
                {
                    optimizer.zero_grad();

                    var prediction = model.forward(data);
                    var output = nll_loss(prediction, target);

                    output.backward();
                    optimizer.step();

                    if (batchId % _logInterval == 0)
                    {
                        Console.WriteLine($"\rTrain: epoch {epoch} [{batchId * batchSize} / {size}] Loss: {output.ToSingle():F4}");
                    }

                    batchId++;
                }
            }
        }

        private static void Test(
            Module<Tensor, Tensor> model,
            TorchSharp.Modules.SummaryWriter writer,
            Device device,
            IEnumerable<(Tensor, Tensor)> dataLoader,
            int epoch,
            int size)
        {
            model.eval();

            double testLoss = 0;
            int correct = 0;

            foreach (var (data, target) in dataLoader)
            {
                using (var d = torch.NewDisposeScope())
                {
                    var prediction = model.forward(data);
                    var output = nll_loss(prediction, target, reduction: Reduction.Sum);
                    testLoss += output.ToSingle();

                    correct += prediction.argmax(1).eq(target).sum().ToInt32();
                }
            }

            Console.WriteLine($"\rTest set: Average loss {(testLoss / size):F4} | Accuracy {((double)correct / size):P2}");

            if (writer != null)
            {
                writer.add_scalar("mnist_rnn/loss", (float)(testLoss / size), epoch);
                writer.add_scalar("mnist_rnn/accuracy", (float)correct / size, epoch);
            }
        }
    }
}
