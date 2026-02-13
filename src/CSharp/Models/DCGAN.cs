// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharp.Examples
{
    /// <summary>
    /// DCGAN Generator model.
    ///
    /// Based on: https://github.com/pytorch/examples/tree/main/dcgan
    /// </summary>
    public class DCGANGenerator : Module<Tensor, Tensor>
    {
        private Module<Tensor, Tensor> main;

        public DCGANGenerator(string name, int nz, int ngf, int nc, torch.Device device = null) : base(name)
        {
            main = Sequential(
                // input is Z, going into a convolution
                ConvTranspose2d(nz, ngf * 8, 4, stride: 1, padding: 0, bias: false),
                BatchNorm2d(ngf * 8),
                ReLU(inplace: true),
                // state size: (ngf*8) x 4 x 4
                ConvTranspose2d(ngf * 8, ngf * 4, 4, stride: 2, padding: 1, bias: false),
                BatchNorm2d(ngf * 4),
                ReLU(inplace: true),
                // state size: (ngf*4) x 8 x 8
                ConvTranspose2d(ngf * 4, ngf * 2, 4, stride: 2, padding: 1, bias: false),
                BatchNorm2d(ngf * 2),
                ReLU(inplace: true),
                // state size: (ngf*2) x 16 x 16
                ConvTranspose2d(ngf * 2, ngf, 4, stride: 2, padding: 1, bias: false),
                BatchNorm2d(ngf),
                ReLU(inplace: true),
                // state size: (ngf) x 32 x 32
                ConvTranspose2d(ngf, nc, 4, stride: 2, padding: 1, bias: false),
                Tanh()
                // state size: (nc) x 64 x 64
            );

            RegisterComponents();

            if (device != null && device.type != DeviceType.CPU)
                this.to(device);
        }

        public override Tensor forward(Tensor input)
        {
            return main.forward(input);
        }
    }

    /// <summary>
    /// DCGAN Discriminator model.
    /// </summary>
    public class DCGANDiscriminator : Module<Tensor, Tensor>
    {
        private Module<Tensor, Tensor> main;

        public DCGANDiscriminator(string name, int ndf, int nc, torch.Device device = null) : base(name)
        {
            main = Sequential(
                // input is (nc) x 64 x 64
                Conv2d(nc, ndf, kernel_size: 4, stride: 2, padding: 1, bias: false),
                LeakyReLU(0.2, inplace: true),
                // state size: (ndf) x 32 x 32
                Conv2d(ndf, ndf * 2, kernel_size: 4, stride: 2, padding: 1, bias: false),
                BatchNorm2d(ndf * 2),
                LeakyReLU(0.2, inplace: true),
                // state size: (ndf*2) x 16 x 16
                Conv2d(ndf * 2, ndf * 4, kernel_size: 4, stride: 2, padding: 1, bias: false),
                BatchNorm2d(ndf * 4),
                LeakyReLU(0.2, inplace: true),
                // state size: (ndf*4) x 8 x 8
                Conv2d(ndf * 4, ndf * 8, kernel_size: 4, stride: 2, padding: 1, bias: false),
                BatchNorm2d(ndf * 8),
                LeakyReLU(0.2, inplace: true),
                // state size: (ndf*8) x 4 x 4
                Conv2d(ndf * 8, 1, kernel_size: 4, stride: 1, padding: (long)0, bias: false),
                Sigmoid()
            );

            RegisterComponents();

            if (device != null && device.type != DeviceType.CPU)
                this.to(device);
        }

        public override Tensor forward(Tensor input)
        {
            return main.forward(input).view(-1, 1).squeeze(1);
        }
    }
}
