// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace TorchSharp.Examples
{
    /// <summary>
    /// MNIST RNN model using LSTM.
    ///
    /// Based on: https://github.com/pytorch/examples/tree/main/mnist_rnn
    ///
    /// Treats each 28x28 MNIST image as a sequence of 28 time steps,
    /// each with 28 features, and classifies using an LSTM.
    /// </summary>
    public class MNISTRNNModel : Module<Tensor, Tensor>
    {
        private Modules.LSTM rnn;
        private Modules.BatchNorm1d batchnorm;
        private Module<Tensor, Tensor> dropout1 = Dropout(0.25);
        private Module<Tensor, Tensor> dropout2 = Dropout(0.5);
        private Modules.Linear fc1 = Linear(64, 32);
        private Modules.Linear fc2 = Linear(32, 10);

        public MNISTRNNModel(string name, torch.Device device = null) : base(name)
        {
            rnn = LSTM(inputSize: 28, hiddenSize: 64, batchFirst: true);
            batchnorm = BatchNorm1d(64);

            RegisterComponents();

            if (device != null && device.type != DeviceType.CPU)
                this.to(device);
        }

        public override Tensor forward(Tensor input)
        {
            // Shape of input is (batch_size, 1, 28, 28)
            // Reshape to (batch_size, 28, 28) for LSTM
            var x = input.reshape(-1, 28, 28);

            // LSTM forward
            var (output, _, _) = rnn.forward(x);

            // Get last output of RNN: output[:, -1, :]
            x = output[TensorIndex.Colon, TensorIndex.Single(-1), TensorIndex.Colon];

            x = batchnorm.forward(x);
            x = dropout1.forward(x);
            x = fc1.forward(x);
            x = relu(x);
            x = dropout2.forward(x);
            x = fc2.forward(x);
            x = log_softmax(x, dim: 1);
            return x;
        }
    }
}
