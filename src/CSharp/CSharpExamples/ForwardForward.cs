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
    /// Forward-Forward MNIST classification
    ///
    /// Based on: https://github.com/pytorch/examples/tree/main/mnist_forward_forward
    ///
    /// Implements the Forward-Forward algorithm (Geoffrey Hinton, 2022). Instead of
    /// backpropagation, each layer is trained independently using a local contrastive loss.
    /// Positive examples have the correct label overlaid, negative examples have wrong labels.
    /// </summary>
    public class ForwardForward
    {
        internal static void Run(int epochs, int timeout, string logdir)
        {
            var device =
                torch.cuda.is_available() ? torch.CUDA :
                torch.mps_is_available() ? torch.MPS :
                torch.CPU;

            Console.WriteLine();
            Console.WriteLine($"\tRunning Forward-Forward MNIST on {device.type} for {epochs} epochs.");
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

            Console.WriteLine($"\tLoading data...");

            // Load full training set as a single batch for the Forward-Forward algorithm
            int trainSize = 50000;
            int testSize = 10000;

            using (MNISTReader trainReader = new MNISTReader(targetDir, "train", trainSize, device: device),
                               testReader = new MNISTReader(targetDir, "t10k", testSize, device: device))
            {
                Stopwatch totalTime = new Stopwatch();
                totalTime.Start();

                // Get one big batch of training data
                Tensor x = null, y = null, xTe = null, yTe = null;

                foreach (var (data, target) in trainReader) {
                    // Flatten the images: (N, 1, 28, 28) -> (N, 784)
                    x = data.view(data.shape[0], -1);
                    y = target;
                    break; // Just the first (and only) batch
                }

                foreach (var (data, target) in testReader) {
                    xTe = data.view(data.shape[0], -1);
                    yTe = target;
                    break;
                }

                Console.WriteLine($"\tCreating Forward-Forward network [784, 500, 500]...");

                var net = new ForwardForwardNet(new int[] { 784, 500, 500 }, device);

                // Create positive and negative examples
                var xPos = ForwardForwardNet.OverlayLabelOnInput(x, y);
                var yNeg = ForwardForwardNet.GetNegativeLabels(y);
                var xNeg = ForwardForwardNet.OverlayLabelOnInput(x, yNeg);

                Console.WriteLine($"\tTraining...");
                net.Train(xPos, xNeg, epochs, lr: 0.03, logInterval: 10);

                // Evaluate
                var trainPred = net.Predict(x);
                var trainError = 1.0f - trainPred.eq(y).to_type(ScalarType.Float32).mean().item<float>();
                Console.WriteLine($"\tTrain error: {trainError:F4}");

                var testPred = net.Predict(xTe);
                var testError = 1.0f - testPred.eq(yTe).to_type(ScalarType.Float32).mean().item<float>();
                Console.WriteLine($"\tTest error: {testError:F4}");

                totalTime.Stop();
                Console.WriteLine($"Elapsed time: {totalTime.Elapsed.TotalSeconds:F1} s.");
            }
        }
    }
}
