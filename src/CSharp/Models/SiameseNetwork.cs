// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace TorchSharp.Examples
{
    /// <summary>
    /// Siamese Network model based on: https://github.com/pytorch/examples/tree/main/siamese_network
    ///
    /// Uses two identical sub-networks (ResNet-18 backbone) to compare pairs of images.
    /// The network outputs a similarity score (via sigmoid) between 0 and 1.
    /// Trained with BCELoss on MNIST image pairs.
    /// </summary>
    public class SiameseNetworkModel : Module<Tensor, Tensor, Tensor>
    {
        private Module<Tensor, Tensor> backbone;
        private Module<Tensor, Tensor> fc;
        private Module<Tensor, Tensor> sigmoid = Sigmoid();
        private long fcInFeatures;

        public SiameseNetworkModel(string name, torch.Device device = null) : base(name)
        {
            // Build a simple CNN backbone (similar to a mini ResNet for 28x28 grayscale)
            // We use a simpler backbone since we don't have torchvision.models in TorchSharp examples
            var backboneModules = Sequential(
                ("conv1", Conv2d(1, 32, 3, stride: 2, padding: 1)),
                ("bn1", BatchNorm2d(32)),
                ("relu1", ReLU()),
                ("conv2", Conv2d(32, 64, 3, stride: 2, padding: 1)),
                ("bn2", BatchNorm2d(64)),
                ("relu2", ReLU()),
                ("conv3", Conv2d(64, 128, 3, stride: 2, padding: 1)),
                ("bn3", BatchNorm2d(128)),
                ("relu3", ReLU()),
                ("avgpool", AdaptiveAvgPool2d(1))
            );
            backbone = backboneModules;
            fcInFeatures = 128;

            fc = Sequential(
                ("fc1", Linear(fcInFeatures * 2, 256)),
                ("relu", ReLU(inplace: true)),
                ("fc2", Linear(256, 1))
            );

            RegisterComponents();
            InitWeights();

            if (device != null && device.type != DeviceType.CPU)
                this.to(device);
        }

        private void InitWeights()
        {
            foreach (var (paramName, param) in this.named_parameters()) {
                if (paramName.Contains("weight") && param.dim() >= 2) {
                    init.xavier_uniform_(param);
                } else if (paramName.Contains("bias")) {
                    init.constant_(param, 0.01);
                }
            }
        }

        private Tensor ForwardOnce(Tensor x)
        {
            var output = backbone.forward(x);
            output = output.view(output.shape[0], -1);
            return output;
        }

        public override Tensor forward(Tensor input1, Tensor input2)
        {
            var output1 = ForwardOnce(input1);
            var output2 = ForwardOnce(input2);

            // Concatenate both features
            var combined = torch.cat(new Tensor[] { output1, output2 }, dim: 1);

            var output = fc.forward(combined);
            output = sigmoid.forward(output);
            return output;
        }
    }
}
