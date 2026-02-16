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
    /// Siamese Network for image similarity
    ///
    /// Based on: https://github.com/pytorch/examples/tree/main/siamese_network
    ///
    /// Trains a Siamese network to determine if two MNIST images are from the
    /// same class or different classes. Uses BCELoss for training.
    /// </summary>
    public class SiameseNetwork
    {
        private static int _trainBatchSize = 64;
        private static int _testBatchSize = 128;
        private readonly static int _logInterval = 100;

        internal static void Run(int epochs, int timeout, string logdir)
        {
            var device =
                torch.cuda.is_available() ? torch.CUDA :
                torch.mps_is_available() ? torch.MPS :
                torch.CPU;

            Console.WriteLine();
            Console.WriteLine($"\tRunning Siamese Network on {device.type} for {epochs} epochs, terminating after {TimeSpan.FromSeconds(timeout)}.");
            Console.WriteLine();

            torch.random.manual_seed(1);

            var dataset = "mnist";
            var datasetPath = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", dataset);

            var sourceDir = datasetPath;
            var targetDir = Path.Combine(datasetPath, "test_data");

            if (!Directory.Exists(targetDir)) {
                Directory.CreateDirectory(targetDir);
                Decompress.DecompressGZipFile(Path.Combine(sourceDir, "train-images-idx3-ubyte.gz"), targetDir);
                Decompress.DecompressGZipFile(Path.Combine(sourceDir, "train-labels-idx1-ubyte.gz"), targetDir);
                Decompress.DecompressGZipFile(Path.Combine(sourceDir, "t10k-images-idx3-ubyte.gz"), targetDir);
                Decompress.DecompressGZipFile(Path.Combine(sourceDir, "t10k-labels-idx1-ubyte.gz"), targetDir);
            }

            Console.WriteLine($"\tCreating the model...");

            var model = new SiameseNetworkModel("siamese", device);
            var optimizer = optim.Adadelta(model.parameters(), lr: 1.0);
            var scheduler = optim.lr_scheduler.StepLR(optimizer, 1, 0.7);

            Console.WriteLine($"\tPreparing training and test data...");
            Console.WriteLine();

            using (MNISTReader train = new MNISTReader(targetDir, "train", _trainBatchSize, device: device, shuffle: true),
                               test = new MNISTReader(targetDir, "t10k", _testBatchSize, device: device))
            {
                Stopwatch totalTime = new Stopwatch();
                totalTime.Start();

                for (var epoch = 1; epoch <= epochs; epoch++) {
                    Train(model, optimizer, device, train, epoch, train.Size);
                    Test(model, device, test, epoch, test.Size);
                    scheduler.step();

                    Console.WriteLine($"End-of-epoch memory use: {GC.GetTotalMemory(false)}");

                    if (totalTime.Elapsed.TotalSeconds > timeout) break;
                }

                totalTime.Stop();
                Console.WriteLine($"Elapsed time: {totalTime.Elapsed.TotalSeconds:F1} s.");
            }
        }

        /// <summary>
        /// Creates pairs of images from the same dataset for Siamese training.
        /// Even indices create same-class pairs (label=1), odd create different-class pairs (label=0).
        /// </summary>
        private static (Tensor, Tensor, Tensor) CreatePairs(Tensor data, Tensor labels, int batchIdx)
        {
            var rng = new Random(batchIdx);
            int batchSize = (int)data.shape[0];

            var images1 = new List<Tensor>();
            var images2 = new List<Tensor>();
            var targets = new List<float>();

            for (int i = 0; i < batchSize; i++) {
                images1.Add(data[i].unsqueeze(0));

                if (i % 2 == 0) {
                    // Same class pair
                    var sameLabel = labels[i].item<long>();
                    // Find another image with the same label
                    int j = rng.Next(batchSize);
                    int attempts = 0;
                    while (labels[j].item<long>() != sameLabel && attempts < batchSize) {
                        j = rng.Next(batchSize);
                        attempts++;
                    }
                    images2.Add(data[j].unsqueeze(0));
                    targets.Add(1.0f);
                } else {
                    // Different class pair
                    var thisLabel = labels[i].item<long>();
                    int j = rng.Next(batchSize);
                    int attempts = 0;
                    while (labels[j].item<long>() == thisLabel && attempts < batchSize) {
                        j = rng.Next(batchSize);
                        attempts++;
                    }
                    images2.Add(data[j].unsqueeze(0));
                    targets.Add(0.0f);
                }
            }

            var img1 = torch.cat(images1.ToArray(), dim: 0);
            var img2 = torch.cat(images2.ToArray(), dim: 0);
            var tgt = torch.tensor(targets.ToArray());

            return (img1, img2, tgt);
        }

        private static void Train(
            SiameseNetworkModel model,
            optim.Optimizer optimizer,
            Device device,
            IEnumerable<(Tensor, Tensor)> dataLoader,
            int epoch,
            int size)
        {
            model.train();
            var criterion = BCELoss();
            int batchIdx = 0;

            foreach (var (data, labels) in dataLoader) {
                using (var d = torch.NewDisposeScope()) {
                    var (images1, images2, targets) = CreatePairs(data, labels, batchIdx);
                    targets = targets.to(device);

                    optimizer.zero_grad();
                    var outputs = model.forward(images1, images2).squeeze();
                    var loss = criterion.forward(outputs, targets);
                    loss.backward();
                    optimizer.step();

                    if (batchIdx % _logInterval == 0) {
                        Console.WriteLine($"\tTrain Epoch: {epoch} [{batchIdx * _trainBatchSize}/{size}] Loss: {loss.item<float>():F6}");
                    }
                    batchIdx++;
                }
            }
        }

        private static void Test(
            SiameseNetworkModel model,
            Device device,
            IEnumerable<(Tensor, Tensor)> dataLoader,
            int epoch,
            int size)
        {
            model.eval();
            double testLoss = 0;
            int correct = 0;
            int total = 0;
            var criterion = BCELoss();

            using (torch.no_grad()) {
                int batchIdx = 0;
                foreach (var (data, labels) in dataLoader) {
                    using (var d = torch.NewDisposeScope()) {
                        var (images1, images2, targets) = CreatePairs(data, labels, batchIdx + 10000);
                        targets = targets.to(device);

                        var outputs = model.forward(images1, images2).squeeze();
                        testLoss += criterion.forward(outputs, targets).item<float>();

                        var pred = torch.where(outputs > 0.5, 1, 0);
                        correct += pred.eq(targets.to_type(ScalarType.Int32).view_as(pred)).sum().item<int>();
                        total += (int)targets.shape[0];
                        batchIdx++;
                    }
                }
            }

            Console.WriteLine($"====> Test set: Average loss: {testLoss / total:F4}, Accuracy: {correct}/{total} ({100.0 * correct / total:F0}%)");
        }
    }
}
