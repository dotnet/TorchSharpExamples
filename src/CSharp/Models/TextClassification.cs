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
    /// https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
    ///
    /// </summary>
    public class TextClassificationModel : Module<Tensor, Tensor, Tensor>
    {
        private Modules.EmbeddingBag embedding;
        private Modules.Linear fc;

        public TextClassificationModel(long vocab_size, long embed_dim, long num_class) : base("TextClassification")
        {
            embedding = EmbeddingBag(vocab_size, embed_dim, sparse: false);
            fc = Linear(embed_dim, num_class);
            InitWeights();

            RegisterComponents();
        }

        private void InitWeights()
        {
            var initrange = 0.5;

            init.uniform_(embedding.weight, -initrange, initrange);
            init.uniform_(fc.weight, -initrange, initrange);
            init.zeros_(fc.bias);
        }

        public override Tensor forward(Tensor input, Tensor offsets)
        {
            var t = embedding.call(input, offsets);
            return fc.forward(t);
        }
    }
}
