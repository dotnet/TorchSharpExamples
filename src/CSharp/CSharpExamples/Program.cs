using System;
using System.IO;
using System.Reflection;
using TorchSharp.Examples.Utils;

namespace CSharpExamples
{
    public class Program
    {
        static void Main(string[] args)
        {
            var argumentsPath = Path.Combine(Path.GetDirectoryName(Assembly.GetEntryAssembly().Location), "arguments.json");
            var argumentParser = new ArgumentParser(new FileInfo(argumentsPath), args);

            if (argumentParser.Count == 0)
            {
                argumentParser.UsingMessage("CSharpExamples", "<model-name>");
                return;
            }

            argumentParser.TryGetValue("epochs", out int epochs, 16);
            argumentParser.TryGetValue("timeout", out int timeout, 3600);
            argumentParser.TryGetValue("logdir", out string logdir, null);

            for (var idx = 0; idx < argumentParser.Count; idx++)
            {
                switch(argumentParser[idx].ToLower())
                {
                    case "mnist":
                    case "fashion-mnist":
                        MNIST.Run(epochs, timeout, logdir, argumentParser[idx].ToLower());
                        break;

                    case "fgsm":
                    case "fashion-fgsm":
                        AdversarialExampleGeneration.Run(epochs, timeout, logdir, argumentParser[idx].ToLower());
                        break;

                    case "alexnet":
                    case "resnet":
                    case "mobilenet":
                    case "resnet18":
                    case "resnet34":
                    case "resnet50":
#if false
            // The following are disabled, because they require big CUDA processors in order to run.
                    case "resnet101":
                    case "resnet152":
#endif
                    case "vgg11":
                    case "vgg13":
                    case "vgg16":
                    case "vgg19":
                        CIFAR10.Run(epochs, timeout, logdir, argumentParser[idx]);
                        break;

                    case "text":
                        TextClassification.Run(epochs, timeout, logdir);
                        break;

                    case "seq2seq":
                        SequenceToSequence.Run(epochs, timeout, logdir);
                        break;

                    default:
                        Console.Error.WriteLine($"Unknown model name: {argumentParser[idx]}");
                        break;
                }
            }
        }
    }
}
