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

namespace CSharpExamples
{
    /// <summary>
    /// Variational Auto-Encoder (VAE)
    ///
    /// Based on: https://github.com/pytorch/examples/tree/main/vae
    ///
    /// Trains a VAE on MNIST data. The model learns to encode images into a
    /// latent space and decode them back, using the reparameterization trick.
    /// </summary>
    /// <remarks>
    /// Uses the same MNIST dataset as the MNIST example.
    /// Download from: http://yann.lecun.com/exdb/mnist/
    /// </remarks>
    public class VAE
    {
        private static int _trainBatchSize = 128;
        private static int _testBatchSize = 128;

        private readonly static int _logInterval = 10;

        internal static void Run(int epochs, int timeout, string logdir)
        {
            var device =
                torch.cuda.is_available() ? torch.CUDA :
                torch.mps_is_available() ? torch.MPS :
                torch.CPU;

            Console.WriteLine();
            Console.WriteLine($"\tRunning VAE on {device.type.ToString()} for {epochs} epochs, terminating after {TimeSpan.FromSeconds(timeout)}.");
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

            var model = new VAEModel("vae", device);

            var optimizer = torch.optim.Adam(model.parameters(), lr: 1e-3);

            Console.WriteLine($"\tPreparing training and test data...");
            Console.WriteLine();

            using (MNISTReader train = new MNISTReader(targetDir, "train", _trainBatchSize, device: device, shuffle: true),
                               test = new MNISTReader(targetDir, "t10k", _testBatchSize, device: device))
            {
                var writer = String.IsNullOrEmpty(logdir) ? null : torch.utils.tensorboard.SummaryWriter(logdir, createRunName: true);

                Stopwatch totalTime = new Stopwatch();
                totalTime.Start();

                for (var epoch = 1; epoch <= epochs; epoch++)
                {
                    Train(model, optimizer, device, train, epoch, train.Size);
                    Test(model, writer, device, test, epoch, test.Size);

                    Console.WriteLine($"End-of-epoch memory use: {GC.GetTotalMemory(false)}");

                    if (totalTime.Elapsed.TotalSeconds > timeout) break;
                }

                totalTime.Stop();
                Console.WriteLine($"Elapsed time: {totalTime.Elapsed.TotalSeconds:F1} s.");
            }
        }

        private static void Train(
            VAEModel model,
            torch.optim.Optimizer optimizer,
            Device device,
            IEnumerable<(Tensor, Tensor)> dataLoader,
            int epoch,
            int size)
        {
            model.train();
            double trainLoss = 0;
            int batchIdx = 0;

            foreach (var (data, _) in dataLoader)
            {
                using (var d = torch.NewDisposeScope())
                {
                    optimizer.zero_grad();

                    var (reconBatch, mu, logvar) = model.forward(data);
                    var loss = VAEModel.LossFunction(reconBatch, data, mu, logvar);

                    loss.backward();
                    trainLoss += loss.item<float>();
                    optimizer.step();

                    if (batchIdx % _logInterval == 0)
                    {
                        Console.WriteLine($"\tTrain Epoch: {epoch} [{batchIdx * _trainBatchSize} / {size}] Loss: {loss.item<float>() / data.shape[0]:F6}");
                    }

                    batchIdx++;
                }
            }

            Console.WriteLine($"====> Epoch: {epoch} Average loss: {trainLoss / size:F4}");
        }

        private static void Test(
            VAEModel model,
            TorchSharp.Modules.SummaryWriter writer,
            Device device,
            IEnumerable<(Tensor, Tensor)> dataLoader,
            int epoch,
            int size)
        {
            model.eval();
            double testLoss = 0;

            using (torch.no_grad())
            {
                foreach (var (data, _) in dataLoader)
                {
                    using (var d = torch.NewDisposeScope())
                    {
                        var (reconBatch, mu, logvar) = model.forward(data);
                        testLoss += VAEModel.LossFunction(reconBatch, data, mu, logvar).item<float>();
                    }
                }
            }

            testLoss /= size;
            Console.WriteLine($"====> Test set loss: {testLoss:F4}");

            if (writer != null)
            {
                writer.add_scalar("vae/loss", (float)testLoss, epoch);
            }
        }
    }
}
