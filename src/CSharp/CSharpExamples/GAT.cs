// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Diagnostics;

using TorchSharp;
using TorchSharp.Examples;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace CSharpExamples
{
    /// <summary>
    /// Graph Attention Network (GAT) for node classification
    ///
    /// Based on: https://github.com/pytorch/examples/tree/main/gat
    ///
    /// Implements a 2-layer GAT with multi-head attention for semi-supervised
    /// node classification. Uses synthetic graph data for demonstration.
    /// </summary>
    public class GAT
    {
        internal static void Run(int epochs, int timeout, string logdir)
        {
            var device =
                torch.cuda.is_available() ? torch.CUDA :
                torch.mps_is_available() ? torch.MPS :
                torch.CPU;

            Console.WriteLine();
            Console.WriteLine($"\tRunning GAT on {device.type} for {epochs} epochs, terminating after {TimeSpan.FromSeconds(timeout)}.");
            Console.WriteLine();

            torch.random.manual_seed(13);

            // Synthetic graph data (simulating Cora-like structure)
            int numNodes = 2708;
            int numFeatures = 1433;
            int numClasses = 7;
            int hiddenDim = 64;
            int numHeads = 8;

            Console.WriteLine($"\tGenerating synthetic graph data...");
            Console.WriteLine($"\t  Nodes: {numNodes}, Features: {numFeatures}, Classes: {numClasses}");
            Console.WriteLine($"\t  Hidden: {hiddenDim}, Heads: {numHeads}");

            var features = torch.randn(numNodes, numFeatures, device: device);
            var labels = torch.randint(numClasses, numNodes, device: device);

            // Create adjacency matrix with self-loops
            var adjMat = torch.eye(numNodes, device: device);
            // Add some random edges to simulate graph structure
            var rng = new Random(13);
            int numEdges = 10556;
            for (int e = 0; e < numEdges; e++) {
                int i = rng.Next(numNodes);
                int j = rng.Next(numNodes);
                adjMat[i, j] = 1.0f;
                adjMat[j, i] = 1.0f;
            }

            // Split
            var idx = torch.randperm(numNodes, device: device);
            var idxTrain = idx.slice(0, 1600, numNodes, 1);
            var idxVal = idx.slice(0, 1200, 1600, 1);
            var idxTest = idx.slice(0, 0, 1200, 1);

            Console.WriteLine($"\tCreating GAT model...");

            var model = new GATModel("gat", numFeatures, hiddenDim, numHeads, numClasses,
                concat: false, dropout: 0.6, leakyReluSlope: 0.2, device: device);

            var optimizer = optim.Adam(model.parameters(), lr: 0.005, weight_decay: 5e-4);
            var criterion = NLLLoss();

            Console.WriteLine($"\tTraining...");

            Stopwatch totalTime = new Stopwatch();
            totalTime.Start();

            for (int epoch = 1; epoch <= epochs; epoch++) {
                using (var d = torch.NewDisposeScope()) {
                    model.train();
                    optimizer.zero_grad();

                    var output = model.forward(features, adjMat);
                    var loss = criterion.forward(output.index(idxTrain), labels.index(idxTrain));
                    loss.backward();
                    optimizer.step();

                    if (epoch % 20 == 0 || epoch == 1) {
                        model.eval();
                        using (torch.no_grad()) {
                            var evalOutput = model.forward(features, adjMat);

                            var trainAcc = evalOutput.index(idxTrain).argmax(1)
                                .eq(labels.index(idxTrain)).to_type(ScalarType.Float32).mean().item<float>();
                            var valAcc = evalOutput.index(idxVal).argmax(1)
                                .eq(labels.index(idxVal)).to_type(ScalarType.Float32).mean().item<float>();

                            Console.WriteLine($"\tEpoch {epoch:D4} | Loss: {loss.item<float>():F4} | Train Acc: {trainAcc:F4} | Val Acc: {valAcc:F4}");
                        }
                    }
                }

                if (totalTime.Elapsed.TotalSeconds > timeout) break;
            }

            // Final test
            model.eval();
            using (torch.no_grad()) {
                var testOutput = model.forward(features, adjMat);
                var testAcc = testOutput.index(idxTest).argmax(1)
                    .eq(labels.index(idxTest)).to_type(ScalarType.Float32).mean().item<float>();
                Console.WriteLine($"\tTest accuracy: {testAcc:F4}");
            }

            totalTime.Stop();
            Console.WriteLine($"Elapsed time: {totalTime.Elapsed.TotalSeconds:F1} s.");
        }
    }
}
