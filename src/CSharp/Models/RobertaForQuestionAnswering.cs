// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using TorchSharp.Modules;

namespace TorchSharp.Examples
{
    public class RobertaForQuestionAnswering : torch.nn.Module
    {
        public readonly Roberta bert;
        public readonly Linear qa_outputs;

        public RobertaForQuestionAnswering(int numLayers, int numAttentionHeads,
            long numEmbeddings, long embeddingSize, long hiddenSize, long outputSize, long ffnHiddenSize,
            long maxPositions, long maxTokenTypes, double layerNormEps,
            double embeddingDropoutRate, double attentionDropoutRate, double attentionOutputDropoutRate, double outputDropoutRate)
            : base(nameof(RobertaForQuestionAnswering))
        {
            bert = new Roberta(
                numLayers: numLayers,
                numAttentionHeads: numAttentionHeads,
                numEmbeddings: numEmbeddings,
                embeddingSize: embeddingSize,
                hiddenSize: hiddenSize,
                outputSize: outputSize,
                ffnHiddenSize: ffnHiddenSize,
                maxPositions: maxPositions,
                maxTokenTypes: maxTokenTypes,
                layerNormEps: layerNormEps,
                embeddingDropoutRate: embeddingDropoutRate,
                attentionDropoutRate: attentionDropoutRate,
                attentionOutputDropoutRate: attentionOutputDropoutRate,
                outputDropoutRate: outputDropoutRate);
            qa_outputs = torch.nn.Linear(inputSize: outputSize, outputSize: 2);

            apply(InitWeights);

            RegisterComponents();
        }

        public (torch.Tensor startLogits, torch.Tensor endLogits) forward(
            torch.Tensor tokens, torch.Tensor positions, torch.Tensor tokenTypes, torch.Tensor attentionMask)
        {
            using var disposeScope = torch.NewDisposeScope();
            var encodedVector = bert.forward(tokens, positions, tokenTypes, attentionMask);
            var logits = qa_outputs.forward(encodedVector);
            var splitLogits = logits.split(1, dimension: -1);
            var startLogits = splitLogits[0].squeeze(-1).contiguous();
            var endLogits = splitLogits[1].squeeze(-1).contiguous();
            return (startLogits.MoveToOuterDisposeScope(), endLogits.MoveToOuterDisposeScope());
        }

        private static void InitWeights(torch.nn.Module module)
        {
            using var disposeScope = torch.NewDisposeScope();
            if (module is Linear linearModule)
            {
                linearModule.weight.normal_(mean: 0.0, stddev: 0.02);
                if (linearModule.bias.IsNotNull())
                {
                    linearModule.bias.zero_();
                }
            }
            else if (module is Embedding embeddingModule)
            {
                embeddingModule.weight.normal_(mean: 0.0, stddev: 0.02);
                embeddingModule.weight[1].zero_();  // padding_idx
            }
            else if (module is LayerNorm layerNormModule)
            {
                layerNormModule.weight.fill_(1.0);
                layerNormModule.bias.zero_();
            }
        }
    }
}