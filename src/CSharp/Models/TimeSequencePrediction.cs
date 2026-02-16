// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharp.Examples
{
    /// <summary>
    /// Time sequence prediction model using stacked LSTMCells.
    ///
    /// Based on: https://github.com/pytorch/examples/tree/main/time_sequence_prediction
    ///
    /// Uses two stacked LSTMCells followed by a linear layer to predict
    /// future values of a time sequence (sine waves).
    /// </summary>
    public class SequenceModel : Module<Tensor, int, Tensor>
    {
        private Modules.LSTMCell lstm1;
        private Modules.LSTMCell lstm2;
        private Modules.Linear linear;

        public SequenceModel(string name, torch.Device device = null) : base(name)
        {
            lstm1 = LSTMCell(1, 51);
            lstm2 = LSTMCell(51, 51);
            linear = Linear(51, 1);

            RegisterComponents();

            if (device != null && device.type != DeviceType.CPU)
                this.to(device);
        }

        /// <summary>
        /// Forward pass. Processes the input sequence step by step through two stacked LSTMCells,
        /// then optionally predicts 'future' additional steps using its own output as input.
        /// </summary>
        /// <param name="input">Input tensor of shape (batch_size, sequence_length)</param>
        /// <param name="future">Number of future steps to predict beyond the input</param>
        /// <returns>Output tensor of shape (batch_size, sequence_length + future)</returns>
        public override Tensor forward(Tensor input, int future)
        {
            var outputs = new List<Tensor>();
            var batchSize = input.shape[0];

            // Initialize hidden states and cell states to zeros
            var h_t = torch.zeros(batchSize, 51, dtype: torch.float64, device: input.device);
            var c_t = torch.zeros(batchSize, 51, dtype: torch.float64, device: input.device);
            var h_t2 = torch.zeros(batchSize, 51, dtype: torch.float64, device: input.device);
            var c_t2 = torch.zeros(batchSize, 51, dtype: torch.float64, device: input.device);

            // Process input sequence
            var steps = input.split(1, dim: 1);
            Tensor output = null;
            foreach (var input_t in steps)
            {
                var (h1, c1) = lstm1.forward(input_t, (h_t, c_t));
                h_t = h1;
                c_t = c1;
                var (h2, c2) = lstm2.forward(h_t, (h_t2, c_t2));
                h_t2 = h2;
                c_t2 = c2;
                output = linear.forward(h_t2);
                outputs.Add(output);
            }

            // Predict future steps using own output as input
            for (int i = 0; i < future; i++)
            {
                var (h1, c1) = lstm1.forward(output, (h_t, c_t));
                h_t = h1;
                c_t = c1;
                var (h2, c2) = lstm2.forward(h_t, (h_t2, c_t2));
                h_t2 = h2;
                c_t2 = c2;
                output = linear.forward(h_t2);
                outputs.Add(output);
            }

            return torch.cat(outputs, dim: 1);
        }
    }
}
