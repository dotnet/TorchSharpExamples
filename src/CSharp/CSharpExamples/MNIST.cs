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
    /// Simple MNIST Convolutional model.
    /// </summary>
    /// <remarks>
    /// There are at least two interesting data sets to use with this example:
    /// 
    /// 1. The classic MNIST set of 60000 images of handwritten digits.
    ///
    ///     It is available at: http://yann.lecun.com/exdb/mnist/
    ///     
    /// 2. The 'fashion-mnist' data set, which has the exact same file names and format as MNIST, but is a harder
    ///    data set to train on. It's just as large as MNIST, and has the same 60/10 split of training and test
    ///    data.
    ///    It is available at: https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion
    ///
    /// In each case, there are four .gz files to download. Place them in a folder and then point the '_dataLocation'
    /// constant below at the folder location.
    /// </remarks>
    public class MNIST
    {
        private static int _epochs = 4;
        private static int _trainBatchSize = 64;
        private static int _testBatchSize = 128;

        private readonly static int _logInterval = 100;

        internal static void Run(int epochs, int timeout, string logdir, string dataset)
        {
            _epochs = epochs;

            if (string.IsNullOrEmpty(dataset))
            {
                dataset = "mnist";
            }

            var device = cuda.is_available() ? CUDA : CPU;

            Console.WriteLine();
            Console.WriteLine($"\tRunning MNIST with {dataset} on {device.type.ToString()} for {epochs} epochs, terminating after {TimeSpan.FromSeconds(timeout)}.");
            Console.WriteLine();

            var datasetPath = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", dataset);

            random.manual_seed(1);

            var cwd = Environment.CurrentDirectory;

            var writer = String.IsNullOrEmpty(logdir) ? null : torch.utils.tensorboard.SummaryWriter(logdir, createRunName: true);

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

            var model = new TorchSharp.Examples.MNIST.Model("model", device);

            var normImage = transforms.Normalize(new double[] { 0.1307 }, new double[] { 0.3081 }, device: (Device)device);

            Console.WriteLine($"\tPreparing training and test data...");
            Console.WriteLine();

            using (MNISTReader train = new MNISTReader(targetDir, "train", _trainBatchSize, device: device, shuffle: true, transform: normImage),
                                test = new MNISTReader(targetDir, "t10k", _testBatchSize, device: device, transform: normImage))
            {

                TrainingLoop(dataset, timeout, writer, device, model, train, test);
            }
        }

        internal static void TrainingLoop(string dataset, int timeout, TorchSharp.Modules.SummaryWriter writer, Device device, Module<Tensor, Tensor> model, MNISTReader train, MNISTReader test)
        {
            var optimizer = optim.Adam(model.parameters());

            var scheduler = optim.lr_scheduler.StepLR(optimizer, 1, 0.7);

            Stopwatch totalTime = new Stopwatch();
            totalTime.Start();

            for (var epoch = 1; epoch <= _epochs; epoch++)
            {

                Train(model, optimizer, NLLLoss(reduction: Reduction.Mean), device, train, epoch, train.BatchSize, train.Size);
                Test(model, NLLLoss(reduction: nn.Reduction.Sum), writer, device, test, epoch, test.Size);

                Console.WriteLine($"End-of-epoch memory use: {GC.GetTotalMemory(false)}");

                if (totalTime.Elapsed.TotalSeconds > timeout) break;
            }

            totalTime.Stop();
            Console.WriteLine($"Elapsed time: {totalTime.Elapsed.TotalSeconds:F1} s.");

            Console.WriteLine("Saving model to '{0}'", dataset + ".model.bin");
            model.save(dataset + ".model.bin");
        }

        private static void Train(
            Module<Tensor, Tensor> model,
            optim.Optimizer optimizer,
            Loss<Tensor, Tensor, Tensor> loss,
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
                    var output = loss.forward(prediction, target);

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
            Loss<Tensor, Tensor, Tensor> loss,
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
                    var output = loss.forward(prediction, target);
                    testLoss += output.ToSingle();

                    correct += prediction.argmax(1).eq(target).sum().ToInt32();
                }
            }

            Console.WriteLine($"Size: {size}, Total: {size}");

            Console.WriteLine($"\rTest set: Average loss {(testLoss / size):F4} | Accuracy {((double)correct / size):P2}");

            if (writer != null)
            {
                writer.add_scalar("MNIST/loss", (float)(testLoss / size), epoch);
                writer.add_scalar("MNIST/accuracy", (float)correct / size, epoch);
            }
        }
    }
}
