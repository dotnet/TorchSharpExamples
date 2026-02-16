// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace TorchSharp.Examples
{
    /// <summary>
    /// Graph Attention Layer as described in "Graph Attention Networks" (https://arxiv.org/pdf/1710.10903.pdf).
    ///
    /// Computes attention coefficients for each edge in the graph, then aggregates neighbor features
    /// using these attention weights.
    /// </summary>
    public class GraphAttentionLayer : Module<Tensor, Tensor, Tensor>
    {
        private readonly int nHeads;
        private readonly int nHidden;
        private readonly int outFeatures;
        private readonly bool concat;
        private readonly double dropoutRate;

        private Modules.Parameter W;
        private Modules.Parameter a;
        private Module<Tensor, Tensor> leakyrelu;

        public GraphAttentionLayer(string name, int inFeatures, int outFeatures, int nHeads,
            bool concat = false, double dropout = 0.4, double leakyReluSlope = 0.2) : base(name)
        {
            this.nHeads = nHeads;
            this.concat = concat;
            this.dropoutRate = dropout;
            this.outFeatures = outFeatures;

            if (concat) {
                if (outFeatures % nHeads != 0)
                    throw new ArgumentException("outFeatures must be a multiple of nHeads when concat is true");
                this.nHidden = outFeatures / nHeads;
            } else {
                this.nHidden = outFeatures;
            }

            W = Parameter(torch.empty(inFeatures, this.nHidden * nHeads));
            a = Parameter(torch.empty(nHeads, 2 * this.nHidden, 1));

            leakyrelu = LeakyReLU(leakyReluSlope);

            RegisterComponents();
            ResetParameters();
        }

        private void ResetParameters()
        {
            init.xavier_normal_(W);
            init.xavier_normal_(a);
        }

        private Tensor GetAttentionScores(Tensor hTransformed)
        {
            var sourceScores = torch.matmul(hTransformed, a.index(new TensorIndex[] {
                TensorIndex.Colon, TensorIndex.Slice(null, nHidden), TensorIndex.Colon }));
            var targetScores = torch.matmul(hTransformed, a.index(new TensorIndex[] {
                TensorIndex.Colon, TensorIndex.Slice(nHidden), TensorIndex.Colon }));

            // (n_heads, n_nodes, 1) + (n_heads, 1, n_nodes) = (n_heads, n_nodes, n_nodes)
            var e = sourceScores + targetScores.mT;
            return leakyrelu.forward(e);
        }

        public override Tensor forward(Tensor h, Tensor adjMat)
        {
            long nNodes = h.shape[0];

            // Apply linear transformation: W * h
            var hTransformed = torch.mm(h, W);
            hTransformed = nn.functional.dropout(hTransformed, dropoutRate, training);

            // Reshape to (n_heads, n_nodes, n_hidden)
            hTransformed = hTransformed.view(nNodes, nHeads, nHidden).permute(1, 0, 2);

            // Get attention scores (n_heads, n_nodes, n_nodes)
            var e = GetAttentionScores(hTransformed);

            // Mask non-existent edges
            var connectivityMask = -9e16 * torch.ones_like(e);
            e = torch.where(adjMat > 0, e, connectivityMask);

            // Softmax over rows
            var attention = softmax(e, dim: -1);
            attention = nn.functional.dropout(attention, dropoutRate, training);

            // Weighted average of neighbor features
            var hPrime = torch.matmul(attention, hTransformed);

            if (concat) {
                hPrime = hPrime.permute(1, 0, 2).contiguous().view(nNodes, outFeatures);
            } else {
                hPrime = hPrime.mean(new long[] { 0 });
            }

            return hPrime;
        }
    }

    /// <summary>
    /// Graph Attention Network (GAT) based on: https://github.com/pytorch/examples/tree/main/gat
    ///
    /// Two-layer GAT for semi-supervised node classification.
    /// The first layer uses multi-head attention with ELU activation.
    /// The second layer uses single-head attention with log-softmax output.
    /// </summary>
    public class GATModel : Module<Tensor, Tensor, Tensor>
    {
        private GraphAttentionLayer gat1;
        private GraphAttentionLayer gat2;

        public GATModel(string name, int inFeatures, int nHidden, int nHeads, int numClasses,
            bool concat = false, double dropout = 0.4, double leakyReluSlope = 0.2,
            torch.Device device = null) : base(name)
        {
            gat1 = new GraphAttentionLayer("gat1", inFeatures, nHidden, nHeads,
                concat: concat, dropout: dropout, leakyReluSlope: leakyReluSlope);
            gat2 = new GraphAttentionLayer("gat2", nHidden, numClasses, 1,
                concat: false, dropout: dropout, leakyReluSlope: leakyReluSlope);

            RegisterComponents();

            if (device != null && device.type != DeviceType.CPU)
                this.to(device);
        }

        public override Tensor forward(Tensor inputTensor, Tensor adjMat)
        {
            var x = gat1.forward(inputTensor, adjMat);
            x = elu(x, 1.0);
            x = gat2.forward(x, adjMat);
            return log_softmax(x, dim: 1);
        }
    }
}
