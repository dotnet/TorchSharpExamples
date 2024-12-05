// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.IO;
using System.Collections.Generic;

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
    /// FGSM Attack
    ///
    /// Based on : https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
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
    ///
    /// The example is based on the PyTorch tutorial, but the results from attacking the model are very different from
    /// what the tutorial article notes, at least on the machine where it was developed. There is an order-of-magnitude lower
    /// drop-off in accuracy in this version. That said, when running the PyTorch tutorial on the same machine, the
    /// accuracy trajectories are the same between .NET and Python. If the base convulutational model is trained
    /// using Python, and then used for the FGSM attack in both .NET and Python, the drop-off trajectories are extremenly
    /// close.
    /// </remarks>
    public class AdversarialExampleGeneration
    {
        private readonly static string _dataLocation = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", "mnist");

        private static int _epochs = 4;
        private static int _trainBatchSize = 64;
        private static int _testBatchSize = 128;

        static internal void Run(int epochs, int timeout, string logdir, string dataset)
        {
            _epochs = epochs;

            if (string.IsNullOrEmpty(dataset))
            {
                dataset = "mnist";
            }

            var cwd = Environment.CurrentDirectory;

            var datasetPath = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", dataset);

            var _ = torch.random.manual_seed(1);

            //var device = torch.CPU;
            var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            Console.WriteLine();
            Console.WriteLine($"\tRunning FGSM attack with {dataset} on {device.type.ToString()} for {epochs} epochs, terminating after {TimeSpan.FromSeconds(timeout)}.");
            Console.WriteLine();

            if (device.type == DeviceType.CUDA) {
                _trainBatchSize *= 4;
                _testBatchSize *= 4;
                _epochs *= 4;
            }

            Console.WriteLine($"\tPreparing training and test data...");

            var sourceDir = _dataLocation;
            var targetDir = Path.Combine(_dataLocation, "test_data");

            var writer = String.IsNullOrEmpty(logdir) ? null : torch.utils.tensorboard.SummaryWriter(logdir, createRunName:true);

            if (!Directory.Exists(targetDir)) {
                Directory.CreateDirectory(targetDir);
                Decompress.DecompressGZipFile(Path.Combine(sourceDir, "train-images-idx3-ubyte.gz"), targetDir);
                Decompress.DecompressGZipFile(Path.Combine(sourceDir, "train-labels-idx1-ubyte.gz"), targetDir);
                Decompress.DecompressGZipFile(Path.Combine(sourceDir, "t10k-images-idx3-ubyte.gz"), targetDir);
                Decompress.DecompressGZipFile(Path.Combine(sourceDir, "t10k-labels-idx1-ubyte.gz"), targetDir);
            }

            TorchSharp.Examples.MNIST.Model model = null;

            var normImage = transforms.Normalize(new double[] { 0.1307 }, new double[] { 0.3081 }, device: (Device)device);

            using (var test = new MNISTReader(targetDir, "t10k", _testBatchSize, device: device, transform: normImage)) {

                var modelFile = dataset + ".model.bin";

                if (!File.Exists(modelFile)) {
                    // We need the model to be trained first, because we want to start with a trained model.
                    Console.WriteLine($"\n  Running MNIST on {device.type.ToString()} in order to pre-train the model.");

                    model = new TorchSharp.Examples.MNIST.Model("model", device);

                    using (var train = new MNISTReader(targetDir, "train", _trainBatchSize, device: device, shuffle: true, transform: normImage)) {
                        MNIST.TrainingLoop(dataset, timeout, writer, (Device)device, model, train, test);
                    }

                    Console.WriteLine("Moving on to the Adversarial model.\n");

                } else {
                    model = new TorchSharp.Examples.MNIST.Model("model", torch.CPU);
                    model.load(modelFile);
                }

                model.to((Device)device);
                model.eval();

                var epsilons = new double[] { 0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50 };

                foreach (var ε in epsilons) {
                    var attacked = Test(model, NLLLoss(), ε, test, test.Size);
                    Console.WriteLine($"Epsilon: {ε:F2}, accuracy: {attacked:P2}");
                }
            }
        }

        private static Tensor Attack(Tensor image, double ε, Tensor data_grad)
        {
            using (var sign = data_grad.sign()) {
                var perturbed = (image + ε * sign).clamp(0.0, 1.0);
                return perturbed;
            }
        }

        private static double Test(
            TorchSharp.Examples.MNIST.Model model,
            Loss<Tensor, Tensor, Tensor> criterion,
            double ε,
            IEnumerable<(Tensor, Tensor)> dataLoader,
            long size)
        {
            int correct = 0;

            foreach (var (data, target) in dataLoader) {

                using (var d = torch.NewDisposeScope())
                {
                    data.requires_grad = true;

                    using (var output = model.forward(data))
                    using (var loss = criterion.forward(output, target))
                    {

                        model.zero_grad();
                        loss.backward();

                        var perturbed = Attack(data, ε, data.grad);

                        using (var final = model.forward(perturbed))
                        {

                            correct += final.argmax(1).eq(target).sum().ToInt32();
                        }
                    }
                }
            }

            return (double)correct / size;
        }
    }
}
