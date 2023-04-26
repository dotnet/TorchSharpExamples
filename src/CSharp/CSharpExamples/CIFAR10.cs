// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.IO;
using System.Linq;
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
    /// Driver for various models trained and evaluated on the CIFAR10 small (32x32) color image data set.
    /// </summary>
    /// <remarks>
    /// The dataset for this example can be found at: https://www.cs.toronto.edu/~kriz/cifar.html
    /// Download the binary file, and place it in a dedicated folder, e.g. 'CIFAR10,' then edit
    /// the '_dataLocation' definition below to point at the right folder.
    ///
    /// Note: so far, CIFAR10 is supported, but not CIFAR100.
    /// </remarks>
    class CIFAR10
    {
        private readonly static string _dataset = "CIFAR10";
        private readonly static string _dataLocation = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", _dataset);

        private static int _trainBatchSize = 64;
        private static int _testBatchSize = 128;

        private readonly static int _logInterval = 25;
        private readonly static int _numClasses = 10;

        internal static void Run(int epochs, int timeout, string logdir, string modelName)
        {
            torch.random.manual_seed(1);

            var device =
                // This worked on a GeForce RTX 2080 SUPER with 8GB, for all the available network architectures.
                // It may not fit with less memory than that, but it's worth modifying the batch size to fit in memory.
                torch.cuda.is_available() ? torch.CUDA :
                torch.CPU;

            if (device.type == DeviceType.CUDA)
            {
                _trainBatchSize *= 8;
                _testBatchSize *= 8;
            }

            Console.WriteLine();
            Console.WriteLine($"\tRunning {modelName} with {_dataset} on {device.type.ToString()} for {epochs} epochs, terminating after {TimeSpan.FromSeconds(timeout)}.");
            Console.WriteLine();

            var writer = String.IsNullOrEmpty(logdir) ? null : torch.utils.tensorboard.SummaryWriter(logdir, createRunName: true);

            var sourceDir = _dataLocation;
            var targetDir = Path.Combine(_dataLocation, "test_data");

            if (!Directory.Exists(targetDir))
            {
                Directory.CreateDirectory(targetDir);
                Decompress.ExtractTGZ(Path.Combine(sourceDir, "cifar-10-binary.tar.gz"), targetDir);
            }

            Console.WriteLine($"\tCreating the model...");

            Module<Tensor,Tensor> model = null;

            switch (modelName.ToLower())
            {
                case "alexnet":
                    model = new AlexNet(modelName, _numClasses, device);
                    break;
                case "mobilenet":
                    model = new MobileNet(modelName, _numClasses, device);
                    break;
                case "vgg11":
                case "vgg13":
                case "vgg16":
                case "vgg19":
                    model = new VGG(modelName, _numClasses, device);
                    break;
                case "resnet18":
                    model = ResNet.ResNet18(_numClasses, device);
                    break;
                case "resnet34":
                    _testBatchSize /= 4;
                    model = ResNet.ResNet34(_numClasses, device);
                    break;
                case "resnet50":
                    _trainBatchSize /= 6;
                    _testBatchSize /= 8;
                    model = ResNet.ResNet50(_numClasses, device);
                    break;
                case "resnet101":
                    _trainBatchSize /= 6;
                    _testBatchSize /= 8;
                    model = ResNet.ResNet101(_numClasses, device);
                    break;
                case "resnet152":
                    _testBatchSize /= 4;
                    model = ResNet.ResNet152(_numClasses, device);
                    break;
            }

            var hflip = transforms.HorizontalFlip();
            var gray = transforms.Grayscale(3);
            var rotate = transforms.Rotate(90);
            var contrast = transforms.AdjustContrast(1.25);

            Console.WriteLine($"\tPreparing training and test data...");
            Console.WriteLine();

            using (var train = new CIFARReader(targetDir, false, _trainBatchSize, shuffle: true, device: device, transforms: new ITransform[] { }))
            using (var test = new CIFARReader(targetDir, true, _testBatchSize, device: device))
            using (var optimizer = torch.optim.Adam(model.parameters(), 0.001))
            {

                Stopwatch totalSW = new Stopwatch();
                totalSW.Start();

                for (var epoch = 1; epoch <= epochs; epoch++)
                {

                    Stopwatch epchSW = new Stopwatch();
                    epchSW.Start();

                    var loss = NLLLoss();

                    Train(model, optimizer, loss, train.Data(), epoch, _trainBatchSize, train.Size);
                    Test(model, loss, writer, modelName.ToLower(), test.Data(), epoch, test.Size);

                    epchSW.Stop();
                    Console.WriteLine($"Elapsed time for this epoch: {epchSW.Elapsed.TotalSeconds} s.");

                    if (totalSW.Elapsed.TotalSeconds > timeout) break;
                }

                totalSW.Stop();
                Console.WriteLine($"Elapsed training time: {totalSW.Elapsed} s.");
            }

            model.Dispose();
        }

        private static void Train(
            Module<Tensor,Tensor> model,
            torch.optim.Optimizer optimizer,
            Loss<Tensor, Tensor, Tensor> loss,
            IEnumerable<(Tensor, Tensor)> dataLoader,
            int epoch,
            long batchSize,
            long size)
        {
            model.train();

            int batchId = 1;
            long total = 0;
            long correct = 0;

            Console.WriteLine($"Epoch: {epoch}...");

            foreach (var (data, target) in dataLoader)
            {

                using (var d = torch.NewDisposeScope())
                {
                    optimizer.zero_grad();

                    var prediction = model.forward(data);
                    var lsm = log_softmax(prediction, 1);
                    var output = loss.forward(lsm, target);

                    output.backward();

                    optimizer.step();

                    total += target.shape[0];

                    correct += prediction.argmax(1).eq(target).sum().ToInt64();

                    if (batchId % _logInterval == 0)
                    {
                        var count = Math.Min(batchId * batchSize, size);
                        Console.WriteLine($"\rTrain: epoch {epoch} [{count} / {size}] Loss: {output.ToSingle().ToString("0.000000")} | Accuracy: { ((float)correct / total).ToString("0.000000") }");
                    }

                    batchId++;
                }
            }
        }

        private static void Test(
            Module<Tensor, Tensor> model,
            Loss<Tensor, Tensor, Tensor> loss,
            TorchSharp.Modules.SummaryWriter writer,
            string modelName,
            IEnumerable<(Tensor, Tensor)> dataLoader,
            int epoch,
            long size)
        {
            model.eval();

            double testLoss = 0;
            long correct = 0;
            int batchCount = 0;

            foreach (var (data, target) in dataLoader)
            {

                using (var d = torch.NewDisposeScope())
                {
                    var prediction = model.forward(data);
                    var lsm = log_softmax(prediction, 1);
                    var output = loss.forward(lsm, target);

                    testLoss += output.ToSingle();
                    batchCount += 1;

                    correct += prediction.argmax(1).eq(target).sum().ToInt64();
                }
            }

            Console.WriteLine($"\rTest set: Average loss {(testLoss / batchCount).ToString("0.0000")} | Accuracy {((float)correct / size).ToString("0.0000")}");

            if (writer != null)
            {
                writer.add_scalar($"{modelName}/loss", (float)(testLoss / batchCount), epoch);
                writer.add_scalar($"{modelName}/accuracy", (float)correct / size, epoch);
            }
        }
    }
}
