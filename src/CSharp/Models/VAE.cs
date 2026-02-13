// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace TorchSharp.Examples
{
    /// <summary>
    /// VAE model based on: https://github.com/pytorch/examples/tree/main/vae
    ///
    /// Variational Auto-Encoder for MNIST.
    /// The encoder maps 784-dim input to a 20-dim latent space.
    /// The decoder maps from 20-dim latent space back to 784-dim output.
    /// </summary>
    public class VAEModel : Module<Tensor, (Tensor, Tensor, Tensor)>
    {
        private Modules.Linear fc1 = Linear(784, 400);
        private Modules.Linear fc21 = Linear(400, 20);
        private Modules.Linear fc22 = Linear(400, 20);
        private Modules.Linear fc3 = Linear(20, 400);
        private Modules.Linear fc4 = Linear(400, 784);

        public VAEModel(string name, torch.Device device = null) : base(name)
        {
            RegisterComponents();

            if (device != null && device.type != DeviceType.CPU)
                this.to(device);
        }

        public (Tensor mu, Tensor logvar) Encode(Tensor x)
        {
            var h1 = relu(fc1.forward(x));
            return (fc21.forward(h1), fc22.forward(h1));
        }

        public Tensor Reparameterize(Tensor mu, Tensor logvar)
        {
            var std = torch.exp(0.5 * logvar);
            var eps = torch.randn_like(std);
            return mu + eps * std;
        }

        public Tensor Decode(Tensor z)
        {
            var h3 = relu(fc3.forward(z));
            return torch.sigmoid(fc4.forward(h3));
        }

        public override (Tensor, Tensor, Tensor) forward(Tensor input)
        {
            var x = input.view(-1, 784);
            var (mu, logvar) = Encode(x);
            var z = Reparameterize(mu, logvar);
            return (Decode(z), mu, logvar);
        }

        /// <summary>
        /// Reconstruction + KL divergence losses summed over all elements and batch.
        /// </summary>
        public static Tensor LossFunction(Tensor recon_x, Tensor x, Tensor mu, Tensor logvar)
        {
            var BCE = binary_cross_entropy(recon_x, x.view(-1, 784), reduction: Reduction.Sum);

            // see Appendix B from VAE paper:
            // Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            // 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            var KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp());

            return BCE + KLD;
        }
    }
}
