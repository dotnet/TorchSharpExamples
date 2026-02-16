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
    /// Super-Resolution using ESPCN (Efficient Sub-Pixel Convolutional Neural Network)
    ///
    /// Based on: https://github.com/pytorch/examples/tree/main/super_resolution
    ///
    /// Trains a model to upscale low-resolution images using the sub-pixel convolution
    /// technique (PixelShuffle). Uses MNIST as a simple dataset for demonstration.
    /// </summary>
    public class SuperResolution
    {
        private static int _trainBatchSize = 64;
        private static int _testBatchSize = 64;
        private static int _upscaleFactor = 2;
        private readonly static int _logInterval = 100;

        internal static void Run(int epochs, int timeout, string logdir)
        {
            var device =
                torch.cuda.is_available() ? torch.CUDA :
                torch.mps_is_available() ? torch.MPS :
                torch.CPU;

            Console.WriteLine();
            Console.WriteLine($"\tRunning SuperResolution on {device.type} for {epochs} epochs, terminating after {TimeSpan.FromSeconds(timeout)}.");
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

            var model = new SuperResolutionModel("super_resolution", _upscaleFactor, device);
            var optimizer = optim.Adam(model.parameters(), lr: 1e-3);
            var loss = MSELoss();

            Console.WriteLine($"\tPreparing training and test data...");
            Console.WriteLine();

            using (MNISTReader train = new MNISTReader(targetDir, "train", _trainBatchSize, device: device, shuffle: true),
                               test = new MNISTReader(targetDir, "t10k", _testBatchSize, device: device))
            {
                Stopwatch totalTime = new Stopwatch();
                totalTime.Start();

                for (var epoch = 1; epoch <= epochs; epoch++) {
                    Train(model, optimizer, loss, device, train, epoch, train.Size);
                    Test(model, loss, device, test, epoch, test.Size);

                    Console.WriteLine($"End-of-epoch memory use: {GC.GetTotalMemory(false)}");

                    if (totalTime.Elapsed.TotalSeconds > timeout) break;
                }

                totalTime.Stop();
                Console.WriteLine($"Elapsed time: {totalTime.Elapsed.TotalSeconds:F1} s.");
            }
        }

        private static void Train(
            SuperResolutionModel model,
            optim.Optimizer optimizer,
            Loss<Tensor, Tensor, Tensor> lossFn,
            Device device,
            IEnumerable<(Tensor, Tensor)> dataLoader,
            int epoch,
            int size)
        {
            model.train();
            int batchIdx = 0;

            foreach (var (data, _) in dataLoader) {
                using (var d = torch.NewDisposeScope()) {
                    // Use the original image as target, downsample as input
                    var target = data;
                    // Simple downscale by average pooling, then upscale back
                    var input = avg_pool2d(data, _upscaleFactor);

                    optimizer.zero_grad();
                    var output = model.forward(input);
                    var loss = lossFn.forward(output, target);
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
            SuperResolutionModel model,
            Loss<Tensor, Tensor, Tensor> lossFn,
            Device device,
            IEnumerable<(Tensor, Tensor)> dataLoader,
            int epoch,
            int size)
        {
            model.eval();
            double testLoss = 0;
            int batches = 0;

            using (torch.no_grad()) {
                foreach (var (data, _) in dataLoader) {
                    using (var d = torch.NewDisposeScope()) {
                        var target = data;
                        var input = avg_pool2d(data, _upscaleFactor);
                        var output = model.forward(input);
                        testLoss += lossFn.forward(output, target).item<float>();
                        batches++;
                    }
                }
            }

            Console.WriteLine($"====> Epoch {epoch}: Average test loss: {testLoss / batches:F6}");
        }
    }
}
