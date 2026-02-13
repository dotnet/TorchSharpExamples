// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;

using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace CSharpExamples
{
    /// <summary>
    /// Polynomial Regression
    ///
    /// Based on: https://github.com/pytorch/examples/tree/main/regression
    ///
    /// This example demonstrates how to train a simple linear model to approximate
    /// a polynomial function using smooth L1 loss. The model learns the coefficients
    /// of a polynomial y = w1*x + w2*x^2 + w3*x^3 + w4*x^4 + b.
    /// </summary>
    public class Regression
    {
        private const int POLY_DEGREE = 4;

        internal static void Run(int epochs, int timeout)
        {
            torch.random.manual_seed(1);

            Console.WriteLine();
            Console.WriteLine($"\tRunning Regression (Polynomial Fitting)");
            Console.WriteLine();

            // Create random target polynomial coefficients
            var W_target = torch.randn(POLY_DEGREE, 1) * 5;
            var b_target = torch.randn(1) * 5;

            Console.WriteLine($"\tTarget function: {PolyDesc(W_target.view(-1), b_target)}");

            // Define model: a single linear layer
            var fc = Linear(POLY_DEGREE, 1);

            for (int batchIdx = 1; ; batchIdx++)
            {
                using (var d = torch.NewDisposeScope())
                {
                    // Get batch
                    var (batch_x, batch_y) = GetBatch(W_target, b_target, 32);

                    // Reset gradients
                    fc.zero_grad();

                    // Forward pass
                    var output = smooth_l1_loss(fc.forward(batch_x), batch_y);
                    var loss = output.item<float>();

                    // Backward pass
                    output.backward();

                    // Apply gradients manually
                    using (torch.no_grad())
                    {
                        foreach (var param in fc.parameters())
                        {
                            param.add_(-0.1f * param.grad);
                        }
                    }

                    // Stop criterion
                    if (loss < 1e-3)
                    {
                        Console.WriteLine($"\tLoss: {loss:F6} after {batchIdx} batches");
                        Console.WriteLine($"\tLearned function: {PolyDesc(fc.weight.view(-1), fc.bias)}");
                        Console.WriteLine($"\tActual function:  {PolyDesc(W_target.view(-1), b_target)}");
                        break;
                    }

                    if (batchIdx % 100 == 0)
                    {
                        Console.WriteLine($"\tBatch {batchIdx}, Loss: {loss:F6}");
                    }

                    if (batchIdx > 10000)
                    {
                        Console.WriteLine($"\tDid not converge after {batchIdx} batches. Loss: {loss:F6}");
                        break;
                    }
                }
            }
        }

        private static Tensor MakeFeatures(Tensor x)
        {
            // Builds features: [x, x^2, x^3, x^4]
            x = x.unsqueeze(1);
            var features = new Tensor[POLY_DEGREE];
            for (int i = 0; i < POLY_DEGREE; i++)
            {
                features[i] = x.pow(i + 1);
            }
            return torch.cat(features, 1);
        }

        private static Tensor F(Tensor x, Tensor W_target, Tensor b_target)
        {
            return x.mm(W_target) + b_target.item<float>();
        }

        private static (Tensor, Tensor) GetBatch(Tensor W_target, Tensor b_target, int batchSize = 32)
        {
            var random = torch.randn(batchSize);
            var x = MakeFeatures(random);
            var y = F(x, W_target, b_target);
            return (x, y);
        }

        private static string PolyDesc(Tensor W, Tensor b)
        {
            var result = "y = ";
            for (int i = 0; i < W.shape[0]; i++)
            {
                var w = W[i].item<float>();
                result += $"{w:+0.00} x^{i + 1} ";
            }
            result += $"{b[0].item<float>():+0.00}";
            return result;
        }
    }
}
