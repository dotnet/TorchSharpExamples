// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;

using TorchSharp;

using TorchSharp.Examples;
using TorchSharp.Examples.Utils;

using static TorchSharp.torch;

using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace CSharpExamples
{
    /// <summary>
    /// This example is based on the PyTorch tutorial at:
    /// 
    /// https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
    ///
    /// It relies on the AG_NEWS dataset, which can be downloaded in CSV form at:
    ///
    /// https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv
    ///
    /// Download the two files, and place them in a folder called "AG_NEWS" in
    /// accordance with the file path below (Windows only).
    ///
    /// </summary>
    public class TextClassification
    {
        private const long emsize = 200;

        private const long batch_size = 128;
        private const long eval_batch_size = 128;

        private const int epochs = 15;

        // This path assumes that you're running this on Windows.
        private readonly static string _dataLocation = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", "AG_NEWS");

        internal static void Run(int epochs, int timeout, string logdir)
        {
            torch.random.manual_seed(1);

            var cwd = Environment.CurrentDirectory;

            var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
            Console.WriteLine();
            Console.WriteLine($"\tRunning TextClassification on {device.type.ToString()} for {epochs} epochs, terminating after {TimeSpan.FromSeconds(timeout)}.");
            Console.WriteLine();

            Console.WriteLine($"\tPreparing training and test data...");

            using (var reader = TorchText.Data.AG_NEWSReader.AG_NEWS("train", (Device)device, _dataLocation))
            {

                var dataloader = reader.Enumerate();

                var tokenizer = TorchText.Data.Utils.get_tokenizer("basic_english");

                var counter = new TorchText.Vocab.Counter<string>();
                foreach (var (label, text) in dataloader)
                {
                    counter.update(tokenizer(text));
                }

                var vocab = new TorchText.Vocab.Vocab(counter);


                Console.WriteLine($"\tCreating the model...");
                Console.WriteLine();

                var model = new TextClassificationModel(vocab.Count, emsize, 4).to((Device)device);

                var loss = CrossEntropyLoss();
                var lr = 5.0;
                var optimizer = torch.optim.SGD(model.parameters(), lr);
                var scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.2, last_epoch: 5);

                var totalTime = new Stopwatch();
                totalTime.Start();

                foreach (var epoch in Enumerable.Range(1, epochs))
                {

                    var sw = new Stopwatch();
                    sw.Start();

                    train(epoch, reader.GetBatches(tokenizer, vocab, batch_size), model, loss, optimizer);

                    sw.Stop();

                    Console.WriteLine($"\nEnd of epoch: {epoch} | lr: {optimizer.ParamGroups.First().LearningRate:0.0000} | time: {sw.Elapsed.TotalSeconds:0.0}s\n");
                    scheduler.step();

                    if (totalTime.Elapsed.TotalSeconds > timeout) break;
                }

                totalTime.Stop();

                using (var test_reader = TorchText.Data.AG_NEWSReader.AG_NEWS("test", (Device)device, _dataLocation))
                {

                    var sw = new Stopwatch();
                    sw.Start();

                    var accuracy = evaluate(test_reader.GetBatches(tokenizer, vocab, eval_batch_size), model, loss);

                    sw.Stop();

                    Console.WriteLine($"\nEnd of training: test accuracy: {accuracy:0.00} | eval time: {sw.Elapsed.TotalSeconds:0.0}s\n");
                    scheduler.step();
                }
            }

        }

        static void train(int epoch, IEnumerable<(Tensor, Tensor, Tensor)> train_data, TextClassificationModel model, Loss<Tensor, Tensor, Tensor> criterion, torch.optim.Optimizer optimizer)
        {
            model.train();

            double total_acc = 0.0;
            long total_count = 0;
            long log_interval = 250;

            var batch = 0;

            var batch_count = train_data.Count();

            using (var d = torch.NewDisposeScope())
            {
                foreach (var (labels, texts, offsets) in train_data)
                {

                    optimizer.zero_grad();

                    using (var predicted_labels = model.forward(texts, offsets))
                    {

                        var loss = criterion.forward(predicted_labels, labels);
                        loss.backward();
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5);
                        optimizer.step();

                        total_acc += (predicted_labels.argmax(1) == labels).sum().to(torch.CPU).item<long>();
                        total_count += labels.size(0);
                    }

                    if (batch % log_interval == 0 && batch > 0)
                    {
                        var accuracy = total_acc / total_count;
                        Console.WriteLine($"epoch: {epoch} | batch: {batch} / {batch_count} | accuracy: {accuracy:0.00}");
                    }
                    batch += 1;
                }
            }
        }

        static double evaluate(IEnumerable<(Tensor, Tensor, Tensor)> test_data, TextClassificationModel model, Loss<Tensor, Tensor, Tensor> criterion)
        {
            model.eval();

            double total_acc = 0.0;
            long total_count = 0;

            using (var d = torch.NewDisposeScope())
            {
                foreach (var (labels, texts, offsets) in test_data)
                {

                    using (var predicted_labels = model.forward(texts, offsets))
                    {
                        var loss = criterion.forward(predicted_labels, labels);

                        total_acc += (predicted_labels.argmax(1) == labels).sum().to(torch.CPU).item<long>();
                        total_count += labels.size(0);
                    }
                }

                return total_acc / total_count;
            }
        }
    }
}
