// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace TorchSharp.Examples
{
    /// <summary>
    /// This example is based on the PyTorch tutorial at:
    /// 
    /// https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    ///
    /// </summary>

    public class TransformerModel : Module<Tensor, Tensor, Tensor>
    {
        private Modules.TransformerEncoder transformer_encoder;
        private PositionalEncoding pos_encoder;
        private Modules.Embedding encoder;
        private Modules.Linear decoder;

        private long ninputs;
        private Device device;

        public TransformerModel(long ntokens, long ninputs, long nheads, long nhidden, long nlayers, double dropout = 0.5) : base("Transformer")
        {
            this.ninputs = ninputs;

            pos_encoder = new PositionalEncoding(ninputs, dropout);
            var encoder_layers = TransformerEncoderLayer(ninputs, nheads, nhidden, dropout);
            transformer_encoder = TransformerEncoder(encoder_layers, nlayers);
            encoder = Embedding(ntokens, ninputs);
            decoder = Linear(ninputs, ntokens);
            InitWeights();

            RegisterComponents();
        }

        public Tensor GenerateSquareSubsequentMask(long size)
        {
            var mask = (torch.ones(new long[] { size, size }) == 1).triu().transpose(0, 1);
            return mask.to_type(ScalarType.Float32)
                .masked_fill(mask == 0, float.NegativeInfinity)
                .masked_fill(mask == 1, 0.0f).to(device);
        }

        private void InitWeights()
        {
            var initrange = 0.1;

            init.uniform_(encoder.weight, -initrange, initrange);
            init.zeros_(decoder.bias);
            init.uniform_(decoder.weight, -initrange, initrange);
        }

        public override Tensor forward(Tensor t, Tensor mask)
        {
            using var src = pos_encoder.forward(encoder.forward(t) * MathF.Sqrt(ninputs));
            using var enc = transformer_encoder.call(src, mask);
            return decoder.forward(enc);
        }

        public TransformerModel to(Device device)
        {
            this.to<TransformerModel>(device);
            this.device = device;
            return this;
        }
    }

    class PositionalEncoding : Module<Tensor, Tensor>
    {
        private Module<Tensor, Tensor> dropout;
        private Tensor pe;

        public PositionalEncoding(long dmodel, double dropout, int maxLen = 5000) : base("PositionalEncoding")
        {
            this.dropout = Dropout(dropout);
            var pe = torch.zeros(new long[] { maxLen, dmodel });
            var position = torch.arange(0, maxLen, 1).unsqueeze(1);
            var divTerm = (torch.arange(0, dmodel, 2) * (-Math.Log(10000.0) / dmodel)).exp();
            pe[TensorIndex.Ellipsis, TensorIndex.Slice(0, null, 2)] = (position * divTerm).sin();
            pe[TensorIndex.Ellipsis, TensorIndex.Slice(1, null, 2)] = (position * divTerm).cos();
            this.pe = pe.unsqueeze(0).transpose(0, 1);

            RegisterComponents();
        }

        public override Tensor forward(Tensor t)
        {
            var x = t + pe[TensorIndex.Slice(null, t.shape[0]), TensorIndex.Slice()];
            return dropout.forward(x);
        }
    }
}
