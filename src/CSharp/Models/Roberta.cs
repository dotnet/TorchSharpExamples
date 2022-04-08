// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Linq;
using TorchSharp.Modules;

namespace TorchSharp.Examples
{
    public class Roberta : torch.nn.Module
    {
        private readonly Embeddings embeddings;
        private readonly Encoder encoder;

        public Roberta(int numLayers, int numAttentionHeads,
            long numEmbeddings, long embeddingSize, long hiddenSize, long outputSize, long ffnHiddenSize,
            long maxPositions, long maxTokenTypes, double layerNormEps,
            double embeddingDropoutRate, double attentionDropoutRate, double attentionOutputDropoutRate, double outputDropoutRate)
            : base(nameof(Roberta))
        {
            embeddings = new Embeddings(numEmbeddings, embeddingSize, maxPositions, maxTokenTypes,
                layerNormEps, embeddingDropoutRate);
            encoder = new Encoder(numLayers, numAttentionHeads, embeddingSize, hiddenSize, outputSize, ffnHiddenSize,
                layerNormEps, attentionDropoutRate, attentionOutputDropoutRate, outputDropoutRate);
            RegisterComponents();
        }

        public override torch.Tensor forward(torch.Tensor t)
        {
            throw new NotImplementedException();
        }

        public override torch.Tensor forward(torch.Tensor x, torch.Tensor y)
        {
            throw new NotImplementedException();
        }

        public torch.Tensor forward(torch.Tensor tokens, torch.Tensor positions, torch.Tensor tokenTypes, torch.Tensor attentionMask)
        {
            using var disposeScope = torch.NewDisposeScope();
            var x = embeddings.forward(tokens, positions, tokenTypes);
            var sequenceOutput = encoder.forward(x, attentionMask);
            return sequenceOutput.MoveToOuterDisposeScope();
        }
    }

    internal class Embeddings : torch.nn.Module
    {
        public readonly Embedding word_embeddings;
        public readonly Embedding position_embeddings;
        public readonly Embedding token_type_embeddings;
        public readonly LayerNorm LayerNorm;
        public readonly Dropout dropout;

        public Embeddings(long numEmbeddings, long embeddingSize, long maxPositions, long maxTokenTypes,
            double layerNormEps, double dropoutRate)
            : base(nameof(Embeddings))
        {
            word_embeddings = torch.nn.Embedding(numEmbeddings, embeddingSize, padding_idx: 1);
            position_embeddings = torch.nn.Embedding(maxPositions, embeddingSize);
            token_type_embeddings = torch.nn.Embedding(maxTokenTypes, embeddingSize);
            LayerNorm = torch.nn.LayerNorm(new long[] { embeddingSize }, eps: layerNormEps);
            dropout = torch.nn.Dropout(dropoutRate);

            RegisterComponents();
        }

        public torch.Tensor forward(torch.Tensor tokens, torch.Tensor positions, torch.Tensor segments)
        {
            using var disposeScope = torch.NewDisposeScope();
            var tokenEmbedding = word_embeddings.forward(tokens);
            var positionEmbedding = position_embeddings.forward(positions);
            var tokenTypeEmbedding = token_type_embeddings.forward(segments);
            var embedding = tokenEmbedding + positionEmbedding + tokenTypeEmbedding;
            var output = LayerNorm.forward(embedding);
            output = dropout.forward(output);
            return output.MoveToOuterDisposeScope();
        }
    }

    internal class Encoder : torch.nn.Module
    {
        public readonly int NumLayers;
        public readonly ModuleList layer;

        public Encoder(int numLayers, int numAttentionHeads, long embeddingSize, long hiddenSize, long outputSize, long ffnHiddenSize,
            double layerNormEps, double dropoutRate, double attentionDropoutRate, double outputDropoutRate)
            : base(nameof(Encoder))
        {
            NumLayers = numLayers;
            layer = new ModuleList(Enumerable.Range(0, numLayers)
                .Select(_ => new Layer(numAttentionHeads, hiddenSize, ffnHiddenSize,
                    layerNormEps, dropoutRate, attentionDropoutRate, outputDropoutRate))
                .ToArray());
            RegisterComponents();
        }

        public override torch.Tensor forward(torch.Tensor x, torch.Tensor attentionMask)
        {
            using var disposeScope = torch.NewDisposeScope();
            foreach (var lyr in layer)
            {
                x = lyr.forward(x, attentionMask);
            }
            return x.MoveToOuterDisposeScope();
        }
    }

    internal class Layer : torch.nn.Module
    {
        public readonly Attention attention;
        public readonly Intermediate intermediate;
        public readonly Output output;

        public Layer(int numAttentionHeads, long hiddenSize, long ffnHiddenSize, double layerNormEps,
            double dropoutRate, double attentionDropoutRate, double outputDropoutRate)
            : base(nameof(Layer))
        {
            attention = new Attention(numAttentionHeads, hiddenSize, layerNormEps, attentionDropoutRate, outputDropoutRate);
            intermediate = new Intermediate(hiddenSize, ffnHiddenSize);
            output = new Output(ffnHiddenSize, hiddenSize, dropoutRate);
            RegisterComponents();
        }

        public override torch.Tensor forward(torch.Tensor input, torch.Tensor attentionMask)
        {
            using var disposeScope = torch.NewDisposeScope();
            var attentionOutput = attention.forward(input, attentionMask);
            var intermediateOutput = intermediate.forward(attentionOutput);
            var layerOutput = output.forward(intermediateOutput, attentionOutput);
            return layerOutput.MoveToOuterDisposeScope();
        }
    }

    internal class Attention : torch.nn.Module
    {
        public readonly AttentionSelf self;
        public readonly AttentionOutput output;

        public Attention(int numAttentionHeads, long hiddenSize, double layerNormEps, double attentionDropoutRate, double outputDropoutRate)
            : base(nameof(Attention))
        {
            self = new AttentionSelf(numAttentionHeads, hiddenSize, layerNormEps, attentionDropoutRate);
            output = new AttentionOutput(hiddenSize, layerNormEps, outputDropoutRate);
            RegisterComponents();
        }

        public override torch.Tensor forward(torch.Tensor hiddenStates, torch.Tensor attentionMask)
        {
            using var disposeScope = torch.NewDisposeScope();
            var x = self.forward(hiddenStates, attentionMask);
            x = output.forward(x, hiddenStates);
            return x.MoveToOuterDisposeScope();
        }
    }

    internal class AttentionSelf : torch.nn.Module
    {
        public readonly int NumAttentionHeads;
        public readonly int AttentionHeadSize;

        public readonly Linear query;
        public readonly Linear key;
        public readonly Linear value;
        public readonly Dropout attention_dropout;

        public AttentionSelf(int numAttentionHeads, long hiddenSize, double layerNormEps, double attentionDropoutRate)
            : base(nameof(AttentionSelf))
        {
            NumAttentionHeads = numAttentionHeads;
            AttentionHeadSize = (int)hiddenSize / numAttentionHeads;
            if (NumAttentionHeads * AttentionHeadSize != hiddenSize)
            {
                throw new ArgumentException($"NumAttentionHeads must be a factor of hiddenSize, got {numAttentionHeads} and {hiddenSize}.");
            }

            query = torch.nn.Linear(hiddenSize, hiddenSize, true);
            key = torch.nn.Linear(hiddenSize, hiddenSize, true);
            value = torch.nn.Linear(hiddenSize, hiddenSize, true);
            attention_dropout = torch.nn.Dropout(attentionDropoutRate);

            RegisterComponents();
        }

        public override torch.Tensor forward(torch.Tensor hiddenStates, torch.Tensor attentionMask)
        {
            using var disposeScope = torch.NewDisposeScope();
            var mixedQueryLayer = query.forward(hiddenStates);
            var mixedKeyLayer = key.forward(hiddenStates);
            var mixedValueLayer = value.forward(hiddenStates);

            var queryLayer = TransposeForScores(mixedQueryLayer);
            var keyLayer = TransposeForScores(mixedKeyLayer);
            var valueLayer = TransposeForScores(mixedValueLayer);

            // Attention
            queryLayer.div_(Math.Sqrt(AttentionHeadSize));
            var attentionScores = torch.matmul(queryLayer, keyLayer.transpose_(-1, -2));
            if (attentionMask is not null && !attentionMask.IsInvalid)
            {
                attentionScores.add_(attentionMask);
            }

            var attentionProbs = torch.nn.functional.softmax(attentionScores, dim: -1);
            attentionProbs = attention_dropout.forward(attentionProbs);

            var contextLayer = torch.matmul(attentionProbs, valueLayer);
            contextLayer = contextLayer.permute(0, 2, 1, 3).contiguous();
            var contextShape = contextLayer.shape[..^2].Append(NumAttentionHeads * AttentionHeadSize).ToArray();
            contextLayer = contextLayer.view(contextShape);
            return contextLayer.MoveToOuterDisposeScope();
        }

        /// <summary>
        /// [B x T x C] -> [B x Head x T x C_Head]
        /// </summary>
        private torch.Tensor TransposeForScores(torch.Tensor x)
        {
            using var disposeScope = torch.NewDisposeScope();
            var newShape = x.shape[..^1].Append(NumAttentionHeads).Append(AttentionHeadSize).ToArray();
            x = x.view(newShape);
            x = x.permute(0, 2, 1, 3).contiguous();
            return x.MoveToOuterDisposeScope();
        }
    }

    internal class AttentionOutput : torch.nn.Module
    {
        public readonly Linear dense;
        public readonly Dropout dropout;
        public readonly LayerNorm LayerNorm;

        public AttentionOutput(long hiddenSize, double layerNormEps, double outputDropoutRate)
            : base(nameof(AttentionOutput))
        {
            dense = torch.nn.Linear(hiddenSize, hiddenSize, true);
            dropout = torch.nn.Dropout(outputDropoutRate);
            LayerNorm = torch.nn.LayerNorm(new long[] { hiddenSize });
            RegisterComponents();
        }

        public override torch.Tensor forward(torch.Tensor hiddenStates, torch.Tensor inputTensor)
        {
            using var disposeScope = torch.NewDisposeScope();
            hiddenStates = dense.forward(hiddenStates);
            hiddenStates = dropout.forward(hiddenStates);
            hiddenStates = LayerNorm.forward(hiddenStates + inputTensor);
            return hiddenStates.MoveToOuterDisposeScope();
        }
    }

    internal class Intermediate : torch.nn.Module
    {
        public readonly Linear dense;
        public readonly GELU gelu;

        public Intermediate(long hiddenSize, long ffnHiddenSize) : base(nameof(Intermediate))
        {
            dense = torch.nn.Linear(hiddenSize, ffnHiddenSize, true);
            gelu = torch.nn.GELU();
            RegisterComponents();
        }

        public override torch.Tensor forward(torch.Tensor t)
        {
            using var disposeScope = torch.NewDisposeScope();
            t = dense.forward(t);
            t = gelu.forward(t);
            return t.MoveToOuterDisposeScope();
        }
    }

    internal class Output : torch.nn.Module
    {
        public readonly Linear dense;
        public readonly LayerNorm LayerNorm;
        public readonly Dropout Dropout;

        public Output(long ffnHiddenSize, long hiddenSize, double outputDropoutRate) : base(nameof(Output))
        {
            dense = torch.nn.Linear(ffnHiddenSize, hiddenSize, true);
            Dropout = torch.nn.Dropout(outputDropoutRate);
            LayerNorm = torch.nn.LayerNorm(new long[] { hiddenSize });
            RegisterComponents();
        }

        public override torch.Tensor forward(torch.Tensor hiddenStates, torch.Tensor inputTensor)
        {
            using var disposeScope = torch.NewDisposeScope();
            hiddenStates = dense.forward(hiddenStates);
            hiddenStates = Dropout.forward(hiddenStates);
            hiddenStates = LayerNorm.forward(hiddenStates + inputTensor);
            return hiddenStates.MoveToOuterDisposeScope();
        }
    }
}