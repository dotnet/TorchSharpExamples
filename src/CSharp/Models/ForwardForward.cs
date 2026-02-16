// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace TorchSharp.Examples
{
    /// <summary>
    /// Forward-Forward MNIST model based on: https://github.com/pytorch/examples/tree/main/mnist_forward_forward
    ///
    /// Implements the Forward-Forward algorithm by Geoffrey Hinton.
    /// Instead of backpropagation, each layer is trained independently using a local loss
    /// that encourages high "goodness" for positive examples and low for negative ones.
    /// </summary>
    public class ForwardForwardLayer : Module<Tensor, Tensor>
    {
        private Modules.Linear linear;
        private Module<Tensor, Tensor> relu = ReLU();
        private double threshold;

        public ForwardForwardLayer(string name, int inFeatures, int outFeatures, double threshold = 2.0, torch.Device device = null) : base(name)
        {
            linear = Linear(inFeatures, outFeatures);
            this.threshold = threshold;

            RegisterComponents();

            if (device != null && device.type != DeviceType.CPU)
                this.to(device);
        }

        public override Tensor forward(Tensor x)
        {
            var xDirection = x / (x.norm(1, keepdim: true, p: 2.0f) + 1e-4);
            return relu.forward(torch.mm(xDirection, linear.weight.t()) + linear.bias.unsqueeze(0));
        }

        /// <summary>
        /// Train this layer using the Forward-Forward algorithm.
        /// Returns detached outputs for positive and negative examples to pass to the next layer.
        /// </summary>
        public (Tensor, Tensor) TrainLayer(Tensor xPos, Tensor xNeg, int numEpochs, double lr, int logInterval = 10)
        {
            var opt = optim.Adam(this.parameters(), lr: lr);

            for (int i = 0; i < numEpochs; i++) {
                using var d = torch.NewDisposeScope();

                var gPos = this.forward(xPos).pow(2).mean(new long[] { 1 });
                var gNeg = this.forward(xNeg).pow(2).mean(new long[] { 1 });

                // Loss: log(1 + exp(-gPos + threshold)) + log(1 + exp(gNeg - threshold))
                var loss = torch.log1p(
                    torch.exp(
                        torch.cat(new Tensor[] {
                            -gPos + threshold,
                            gNeg - threshold
                        })
                    )
                ).mean();

                opt.zero_grad();
                loss.backward();
                opt.step();

                if (i % logInterval == 0) {
                    Console.WriteLine($"\t\tLoss: {loss.item<float>():F4}");
                }

                d.DisposeEverythingBut(gPos, gNeg);
            }

            return (this.forward(xPos).detach(), this.forward(xNeg).detach());
        }
    }

    /// <summary>
    /// Forward-Forward network composed of multiple independently-trained layers.
    /// </summary>
    public class ForwardForwardNet
    {
        private List<ForwardForwardLayer> layers = new List<ForwardForwardLayer>();
        private torch.Device device;

        public ForwardForwardNet(int[] dims, torch.Device device = null)
        {
            this.device = device ?? torch.CPU;
            for (int i = 0; i < dims.Length - 1; i++) {
                layers.Add(new ForwardForwardLayer($"ff_layer_{i}", dims[i], dims[i + 1], device: this.device));
            }
        }

        /// <summary>
        /// Overlay label information onto the input data (first 10 pixels).
        /// </summary>
        public static Tensor OverlayLabelOnInput(Tensor x, Tensor y, int numClasses = 10)
        {
            var x_ = x.clone();
            x_[TensorIndex.Colon, TensorIndex.Slice(null, numClasses)] *= 0.0f;
            for (int i = 0; i < x_.shape[0]; i++) {
                x_[i, y[i].item<long>()] = x.max();
            }
            return x_;
        }

        /// <summary>
        /// Generate negative labels (different from the true labels).
        /// </summary>
        public static Tensor GetNegativeLabels(Tensor y)
        {
            var yNeg = y.clone();
            var rng = new Random();
            for (int i = 0; i < y.shape[0]; i++) {
                var trueLabel = y[i].item<long>();
                long newLabel;
                do {
                    newLabel = rng.Next(10);
                } while (newLabel == trueLabel);
                yNeg[i] = torch.tensor(newLabel);
            }
            return yNeg;
        }

        /// <summary>
        /// Train all layers sequentially using the Forward-Forward algorithm.
        /// </summary>
        public void Train(Tensor xPos, Tensor xNeg, int numEpochs, double lr, int logInterval = 10)
        {
            var hPos = xPos;
            var hNeg = xNeg;
            for (int i = 0; i < layers.Count; i++) {
                Console.WriteLine($"\tTraining layer {i}...");
                (hPos, hNeg) = layers[i].TrainLayer(hPos, hNeg, numEpochs, lr, logInterval);
            }
        }

        /// <summary>
        /// Predict by measuring total "goodness" for each possible label.
        /// </summary>
        public Tensor Predict(Tensor x)
        {
            var goodnessList = new List<Tensor>();

            for (int label = 0; label < 10; label++) {
                var h = OverlayLabelOnInput(x, torch.full(x.shape[0], label, dtype: ScalarType.Int64, device: device));
                var goodness = torch.tensor(0.0f, device: device);
                foreach (var layer in layers) {
                    h = layer.forward(h);
                    goodness = goodness + h.pow(2).mean(new long[] { 1 });
                }
                goodnessList.Add(goodness.unsqueeze(1));
            }

            var goodnessPerLabel = torch.cat(goodnessList.ToArray(), 1);
            return goodnessPerLabel.argmax(1);
        }
    }
}
