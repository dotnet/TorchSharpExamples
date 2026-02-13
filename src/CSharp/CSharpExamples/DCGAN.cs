// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Diagnostics;

using TorchSharp;

using TorchSharp.Examples;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace CSharpExamples
{
    /// <summary>
    /// DCGAN - Deep Convolutional Generative Adversarial Network
    ///
    /// Based on: https://github.com/pytorch/examples/tree/main/dcgan
    ///
    /// This example trains a DCGAN using randomly generated fake data,
    /// since it doesn't require any external dataset downloads.
    /// </summary>
    public class DCGAN
    {
        private const int nz = 100;    // Size of latent z vector
        private const int ngf = 64;    // Size of generator feature maps
        private const int ndf = 64;    // Size of discriminator feature maps
        private const int nc = 3;      // Number of channels (RGB)
        private const int imageSize = 64;
        private const int batchSize = 64;

        internal static void Run(int epochs, int timeout, string logdir)
        {
            torch.random.manual_seed(1);

            var device =
                torch.cuda.is_available() ? torch.CUDA :
                torch.mps_is_available() ? torch.MPS :
                torch.CPU;

            Console.WriteLine();
            Console.WriteLine($"\tRunning DCGAN on {device.type.ToString()} for {epochs} epochs, terminating after {TimeSpan.FromSeconds(timeout)}.");
            Console.WriteLine();

            Console.WriteLine($"\tCreating the models...");

            var netG = new DCGANGenerator("generator", nz, ngf, nc, device);
            var netD = new DCGANDiscriminator("discriminator", ndf, nc, device);

            // Apply weights initialization
            WeightsInit(netG);
            WeightsInit(netD);

            var criterion = BCELoss();
            var fixed_noise = torch.randn(batchSize, nz, 1, 1, device: device);

            var optimizerD = torch.optim.Adam(netD.parameters(), lr: 0.0002, beta1: 0.5);
            var optimizerG = torch.optim.Adam(netG.parameters(), lr: 0.0002, beta1: 0.5);

            var writer = String.IsNullOrEmpty(logdir) ? null : torch.utils.tensorboard.SummaryWriter(logdir, createRunName: true);

            Console.WriteLine($"\tTraining with fake random data (no dataset required)...");
            Console.WriteLine();

            // Number of fake data batches per epoch
            int batchesPerEpoch = 100;

            Stopwatch totalTime = new Stopwatch();
            totalTime.Start();

            for (var epoch = 0; epoch < epochs; epoch++)
            {
                for (int i = 0; i < batchesPerEpoch; i++)
                {
                    using (var d = torch.NewDisposeScope())
                    {
                        ////////////////////////////
                        // (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                        ////////////////////////////

                        // Train with real (fake "real" data since we use random data)
                        netD.zero_grad();
                        var real_cpu = torch.randn(batchSize, nc, imageSize, imageSize, device: device);
                        var batch_size_actual = real_cpu.shape[0];
                        var label = torch.full(batch_size_actual, 1.0f, device: device);

                        var output = netD.forward(real_cpu);
                        var errD_real = criterion.forward(output, label);
                        errD_real.backward();
                        var D_x = output.mean().item<float>();

                        // Train with fake
                        var noise = torch.randn(batch_size_actual, nz, 1, 1, device: device);
                        var fake = netG.forward(noise);
                        label.fill_(0.0f);
                        output = netD.forward(fake.detach());
                        var errD_fake = criterion.forward(output, label);
                        errD_fake.backward();
                        var D_G_z1 = output.mean().item<float>();
                        var errD = errD_real + errD_fake;
                        optimizerD.step();

                        ////////////////////////////
                        // (2) Update G network: maximize log(D(G(z)))
                        ////////////////////////////
                        netG.zero_grad();
                        label.fill_(1.0f);
                        output = netD.forward(fake);
                        var errG = criterion.forward(output, label);
                        errG.backward();
                        var D_G_z2 = output.mean().item<float>();
                        optimizerG.step();

                        if (i % 10 == 0)
                        {
                            Console.WriteLine($"[{epoch}/{epochs}][{i}/{batchesPerEpoch}] Loss_D: {errD.item<float>():F4} Loss_G: {errG.item<float>():F4} D(x): {D_x:F4} D(G(z)): {D_G_z1:F4} / {D_G_z2:F4}");
                        }
                    }
                }

                if (writer != null)
                {
                    writer.add_scalar("dcgan/D_x", 0, epoch);
                    writer.add_scalar("dcgan/D_G_z", 0, epoch);
                }

                if (totalTime.Elapsed.TotalSeconds > timeout) break;
            }

            totalTime.Stop();
            Console.WriteLine($"Elapsed time: {totalTime.Elapsed.TotalSeconds:F1} s.");
        }

        private static void WeightsInit(Module model)
        {
            foreach (var (name, module) in model.named_modules())
            {
                var className = module.GetType().Name;
                if (className.Contains("Conv"))
                {
                    foreach (var param in module.parameters())
                    {
                        if (param.requires_grad && param.dim() >= 2)
                        {
                            init.normal_(param, 0.0, 0.02);
                        }
                    }
                }
                else if (className.Contains("BatchNorm"))
                {
                    foreach (var (pname, param) in module.named_parameters())
                    {
                        if (pname.Contains("weight"))
                        {
                            init.normal_(param, 1.0, 0.02);
                        }
                        else if (pname.Contains("bias"))
                        {
                            init.zeros_(param);
                        }
                    }
                }
            }
        }
    }
}
