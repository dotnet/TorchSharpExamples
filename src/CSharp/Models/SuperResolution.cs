// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharp.Examples
{
    /// <summary>
    /// Super-resolution model based on: https://github.com/pytorch/examples/tree/main/super_resolution
    ///
    /// Uses an efficient sub-pixel convolutional neural network (ESPCN) for super-resolution.
    /// The model learns to upscale low-resolution images by a given factor.
    /// </summary>
    public class SuperResolutionModel : Module<Tensor, Tensor>
    {
        private Modules.Conv2d conv1;
        private Modules.Conv2d conv2;
        private Modules.Conv2d conv3;
        private Modules.Conv2d conv4;
        private Module<Tensor, Tensor> pixelShuffle;
        private Module<Tensor, Tensor> relu = ReLU();

        public SuperResolutionModel(string name, int upscaleFactor, torch.Device device = null) : base(name)
        {
            conv1 = Conv2d(1, 64, 5, stride: 1, padding: 2);
            conv2 = Conv2d(64, 64, 3, stride: 1, padding: 1);
            conv3 = Conv2d(64, 32, 3, stride: 1, padding: 1);
            conv4 = Conv2d(32, upscaleFactor * upscaleFactor, 3, stride: 1, padding: 1);
            pixelShuffle = PixelShuffle(upscaleFactor);

            RegisterComponents();
            InitializeWeights();

            if (device != null && device.type != DeviceType.CPU)
                this.to(device);
        }

        private void InitializeWeights()
        {
            init.orthogonal_(conv1.weight, init.calculate_gain(init.NonlinearityType.ReLU));
            init.orthogonal_(conv2.weight, init.calculate_gain(init.NonlinearityType.ReLU));
            init.orthogonal_(conv3.weight, init.calculate_gain(init.NonlinearityType.ReLU));
            init.orthogonal_(conv4.weight);
        }

        public override Tensor forward(Tensor input)
        {
            var x = relu.forward(conv1.forward(input));
            x = relu.forward(conv2.forward(x));
            x = relu.forward(conv3.forward(x));
            x = pixelShuffle.forward(conv4.forward(x));
            return x;
        }
    }
}
