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
    /// Word-level Language Model using RNN (LSTM/GRU/RNN)
    ///
    /// Based on: https://github.com/pytorch/examples/tree/main/word_language_model
    ///
    /// Trains a word-level language model on WikiText-2 using an RNN (LSTM, GRU, or vanilla RNN).
    /// This complements the existing SequenceToSequence example which uses a Transformer.
    ///
    /// WikiText-2 dataset available at:
    /// https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
    /// </summary>
    public class WordLanguageModel
    {
        private readonly static string _dataLocation = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", "wikitext-2-v1");

        private const long emsize = 200;
        private const long nhid = 200;
        private const long nlayers = 2;
        private const double dropout = 0.2;

        private const int batch_size = 20;
        private const int eval_batch_size = 10;
        private const int bptt = 35;

        internal static void Run(string rnnType, int epochs, int timeout, string logdir)
        {
            torch.random.manual_seed(1111);

            var device =
                torch.cuda.is_available() ? torch.CUDA :
                torch.mps_is_available() ? torch.MPS :
                torch.CPU;

            Console.WriteLine();
            Console.WriteLine($"\tRunning WordLanguageModel ({rnnType}) on {device.type.ToString()} for {epochs} epochs, terminating after {TimeSpan.FromSeconds(timeout)}.");
            Console.WriteLine();

            Console.WriteLine($"\tPreparing training and test data...");

            var vocab_iter = TorchText.Datasets.WikiText2("train", _dataLocation);
            var tokenizer = TorchText.Data.Utils.get_tokenizer("basic_english");

            var counter = new TorchText.Vocab.Counter<string>();
            foreach (var item in vocab_iter)
            {
                counter.update(tokenizer(item));
            }

            var vocab = new TorchText.Vocab.Vocab(counter);

            var (train_iter, valid_iter, test_iter) = TorchText.Datasets.WikiText2(_dataLocation);

            var train_data = Batchify(ProcessInput(train_iter, tokenizer, vocab), batch_size).to((Device)device);
            var valid_data = Batchify(ProcessInput(valid_iter, tokenizer, vocab), eval_batch_size).to((Device)device);
            var test_data = Batchify(ProcessInput(test_iter, tokenizer, vocab), eval_batch_size).to((Device)device);

            var ntokens = vocab.Count;

            Console.WriteLine($"\tVocabulary size: {ntokens}");
            Console.WriteLine($"\tCreating the {rnnType} model...");
            Console.WriteLine();

            var model = new RNNModel(rnnType, ntokens, emsize, nhid, nlayers, dropout);
            model.to((Device)device);

            var criterion = NLLLoss();
            var lr = 20.0;

            var writer = String.IsNullOrEmpty(logdir) ? null : torch.utils.tensorboard.SummaryWriter(logdir, createRunName: true);

            var totalTime = new Stopwatch();
            totalTime.Start();

            double? best_val_loss = null;

            for (var epoch = 1; epoch <= epochs; epoch++)
            {
                var sw = new Stopwatch();
                sw.Start();

                Train(epoch, train_data, model, criterion, ntokens, lr, device);

                var val_loss = Evaluate(valid_data, model, criterion, ntokens, device);
                sw.Stop();

                Console.WriteLine($"\nEnd of epoch: {epoch} | lr: {lr:0.00} | time: {sw.Elapsed.TotalSeconds:0.0}s | valid loss: {val_loss:0.00} | valid ppl: {Math.Exp(val_loss):0.00}\n");

                if (writer != null)
                {
                    writer.add_scalar("wlm/valid_loss", (float)val_loss, epoch);
                    writer.add_scalar("wlm/valid_ppl", (float)Math.Exp(val_loss), epoch);
                }

                // Save best model and anneal learning rate
                if (best_val_loss == null || val_loss < best_val_loss.Value)
                {
                    best_val_loss = val_loss;
                }
                else
                {
                    // Anneal the learning rate if no improvement
                    lr /= 4.0;
                }

                if (totalTime.Elapsed.TotalSeconds > timeout) break;
            }

            var test_loss = Evaluate(test_data, model, criterion, ntokens, device);
            totalTime.Stop();

            Console.WriteLine($"\nEnd of training | time: {totalTime.Elapsed.TotalSeconds:0.0}s | test loss: {test_loss:0.00} | test ppl: {Math.Exp(test_loss):0.00}\n");
        }

        private static void Train(int epoch, Tensor train_data, RNNModel model, Loss<Tensor, Tensor, Tensor> criterion, int ntokens, double lr, Device device)
        {
            model.train();
            var total_loss = 0.0f;
            var log_interval = 200;

            var hidden = model.InitHidden(batch_size, device);

            using (var d = torch.NewDisposeScope())
            {
                var batch = 0;

                for (int i = 0; i < train_data.shape[0] - 1; batch++, i += bptt)
                {
                    var (data, targets) = GetBatch(train_data, i);

                    // Detach hidden state from history
                    hidden = hidden.detach();

                    model.zero_grad();

                    var (output, newHidden) = model.forward(data, hidden);
                    hidden = newHidden;

                    var loss = criterion.forward(output.view(-1, ntokens), targets);
                    loss.backward();

                    // Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25);

                    // Manual SGD update (matching PyTorch example default)
                    using (torch.no_grad())
                    {
                        foreach (var p in model.parameters())
                        {
                            p.add_(p.grad, alpha: (float)(-lr));
                        }
                    }

                    total_loss += loss.to(torch.CPU).item<float>();

                    if (batch % log_interval == 0 && batch > 0)
                    {
                        var cur_loss = total_loss / log_interval;
                        Console.WriteLine($"| epoch {epoch,3} | {batch,5}/{train_data.shape[0] / bptt,5} batches | lr {lr:0.00} | loss {cur_loss:0.00} | ppl {Math.Exp(cur_loss):0.00}");
                        total_loss = 0;
                    }

                    d.DisposeEverythingBut(hidden);
                }
            }
        }

        private static double Evaluate(Tensor eval_data, RNNModel model, Loss<Tensor, Tensor, Tensor> criterion, int ntokens, Device device)
        {
            model.eval();

            var total_loss = 0.0f;
            var hidden = model.InitHidden(eval_batch_size, device);

            using (var d = torch.NewDisposeScope())
            {
                var batch = 0;
                for (int i = 0; i < eval_data.shape[0] - 1; batch++, i += bptt)
                {
                    var (data, targets) = GetBatch(eval_data, i);

                    hidden = hidden.detach();

                    var (output, newHidden) = model.forward(data, hidden);
                    hidden = newHidden;

                    var loss = criterion.forward(output.view(-1, ntokens), targets);
                    total_loss += data.shape[0] * loss.to(torch.CPU).item<float>();

                    d.DisposeEverythingBut(hidden);
                }
            }

            return total_loss / eval_data.shape[0];
        }

        static Tensor ProcessInput(IEnumerable<string> iter, Func<string, IEnumerable<string>> tokenizer, TorchText.Vocab.Vocab vocab)
        {
            List<Tensor> data = new List<Tensor>();
            foreach (var item in iter)
            {
                List<long> itemData = new List<long>();
                foreach (var token in tokenizer(item))
                {
                    itemData.Add(vocab[token]);
                }
                data.Add(torch.tensor(itemData.ToArray(), torch.int64));
            }

            var result = torch.cat(data.Where(t => t.NumberOfElements > 0).ToList(), 0);
            return result;
        }

        static Tensor Batchify(Tensor data, int batch_size)
        {
            var nbatch = data.shape[0] / batch_size;
            using var d2 = data.narrow(0, 0, nbatch * batch_size).view(batch_size, -1).t();
            return d2.contiguous();
        }

        static (Tensor, Tensor) GetBatch(Tensor source, int index)
        {
            var len = Math.Min(bptt, (int)(source.shape[0] - 1 - index));
            var data = source[TensorIndex.Slice(index, index + len)];
            var target = source[TensorIndex.Slice(index + 1, index + 1 + len)].reshape(-1);
            return (data, target);
        }
    }
}
