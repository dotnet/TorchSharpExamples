// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchSharp.Examples
{
    /// <summary>
    /// Word-level language model using RNN (LSTM/GRU/RNN).
    ///
    /// Based on: https://github.com/pytorch/examples/tree/main/word_language_model
    ///
    /// Container module with an encoder (embedding), a recurrent module, and a decoder (linear).
    /// Supports LSTM, GRU, RNN_TANH, and RNN_RELU model types.
    /// </summary>
    public class RNNModel : Module<Tensor, Tensor, (Tensor output, Tensor hidden)>
    {
        private Modules.Dropout drop;
        private Modules.Embedding encoder;
        private Modules.Linear decoder;
        private torch.nn.Module<Tensor, Tensor, (Tensor, Tensor)> rnn_gru;
        private Modules.LSTM rnn_lstm;
        private torch.nn.Module<Tensor, Tensor, (Tensor, Tensor)> rnn_plain;

        private string rnn_type;
        private long nhid;
        private long nlayers;

        public RNNModel(string rnn_type, long ntoken, long ninp, long nhid, long nlayers, double dropout = 0.5, bool tie_weights = false) : base("RNNModel")
        {
            this.rnn_type = rnn_type;
            this.nhid = nhid;
            this.nlayers = nlayers;

            drop = Dropout(dropout);
            encoder = Embedding(ntoken, ninp);

            switch (rnn_type)
            {
                case "LSTM":
                    rnn_lstm = LSTM(ninp, nhid, numLayers: nlayers, dropout: dropout);
                    break;
                case "GRU":
                    rnn_gru = GRU(ninp, nhid, numLayers: nlayers, dropout: dropout);
                    break;
                case "RNN_TANH":
                    rnn_plain = RNN(ninp, nhid, numLayers: nlayers, nonLinearity: NonLinearities.Tanh, dropout: dropout);
                    break;
                case "RNN_RELU":
                    rnn_plain = RNN(ninp, nhid, numLayers: nlayers, nonLinearity: NonLinearities.ReLU, dropout: dropout);
                    break;
                default:
                    throw new ArgumentException($"Invalid model type: '{rnn_type}'. Options are: LSTM, GRU, RNN_TANH, RNN_RELU");
            }

            decoder = Linear(nhid, ntoken);

            // Optionally tie weights
            if (tie_weights)
            {
                if (nhid != ninp)
                    throw new ArgumentException("When using the tied flag, nhid must be equal to emsize");
                decoder.weight = encoder.weight;
            }

            InitWeights();
            RegisterComponents();
        }

        private void InitWeights()
        {
            var initrange = 0.1;
            init.uniform_(encoder.weight, -initrange, initrange);
            init.zeros_(decoder.bias);
            init.uniform_(decoder.weight, -initrange, initrange);
        }

        public override (Tensor output, Tensor hidden) forward(Tensor input, Tensor hidden)
        {
            var emb = drop.forward(encoder.forward(input));
            Tensor output;

            switch (rnn_type)
            {
                case "LSTM":
                    // For LSTM, hidden is a concatenation of h and c along dim 0
                    var h = hidden[TensorIndex.Slice(0, nlayers)];
                    var c = hidden[TensorIndex.Slice(nlayers, null)];
                    var (lstm_out, h_n, c_n) = rnn_lstm.forward(emb, (h, c));
                    output = lstm_out;
                    // Concatenate h and c back together
                    hidden = torch.cat(new[] { h_n, c_n }, dim: 0);
                    break;
                case "GRU":
                    var (gru_out, gru_hidden) = rnn_gru.forward(emb, hidden);
                    output = gru_out;
                    hidden = gru_hidden;
                    break;
                default:
                    var (rnn_out, rnn_hidden) = rnn_plain.forward(emb, hidden);
                    output = rnn_out;
                    hidden = rnn_hidden;
                    break;
            }

            output = drop.forward(output);
            var decoded = decoder.forward(output);
            decoded = decoded.view(-1, decoded.shape[decoded.dim() - 1]);
            return (torch.nn.functional.log_softmax(decoded, dim: 1), hidden);
        }

        /// <summary>
        /// Initialize hidden state for the RNN.
        /// For LSTM, returns h and c concatenated along dim 0.
        /// For other RNN types, returns a single hidden state tensor.
        /// </summary>
        public Tensor InitHidden(long batchSize, torch.Device device)
        {
            if (rnn_type == "LSTM")
            {
                var h = torch.zeros(nlayers, batchSize, nhid, device: device);
                var c = torch.zeros(nlayers, batchSize, nhid, device: device);
                return torch.cat(new[] { h, c }, dim: 0);
            }
            else
            {
                return torch.zeros(nlayers, batchSize, nhid, device: device);
            }
        }
    }
}
