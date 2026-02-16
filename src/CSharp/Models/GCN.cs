// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace TorchSharp.Examples
{
    /// <summary>
    /// Graph Convolutional Layer as described in "Semi-Supervised Classification with Graph Convolutional Networks".
    ///
    /// H' = f(D^(-1/2) * A * D^(-1/2) * H * W)
    /// </summary>
    public class GraphConvLayer : Module<Tensor, Tensor, Tensor>
    {
        private Modules.Parameter kernel;
        private Modules.Parameter bias;

        public GraphConvLayer(string name, int inputDim, int outputDim, bool useBias = false) : base(name)
        {
            kernel = Parameter(torch.empty(inputDim, outputDim));
            init.xavier_normal_(kernel);

            if (useBias) {
                bias = Parameter(torch.zeros(outputDim));
            }

            RegisterComponents();
        }

        public override Tensor forward(Tensor inputTensor, Tensor adjMat)
        {
            // Matrix multiplication between input and weight matrix
            var support = torch.mm(inputTensor, kernel);
            // Sparse or dense matrix multiplication between adjacency matrix and support
            var output = torch.mm(adjMat, support);

            if (bias is not null) {
                output = output + bias;
            }

            return output;
        }
    }

    /// <summary>
    /// Graph Convolutional Network (GCN) based on: https://github.com/pytorch/examples/tree/main/gcn
    ///
    /// Two-layer GCN for semi-supervised node classification on graph data.
    /// Uses the Cora citation network dataset.
    /// </summary>
    public class GCNModel : Module<Tensor, Tensor, Tensor>
    {
        private GraphConvLayer gc1;
        private GraphConvLayer gc2;
        private Module<Tensor, Tensor> dropout;

        public GCNModel(string name, int inputDim, int hiddenDim, int outputDim, bool useBias = true, double dropoutP = 0.1, torch.Device device = null) : base(name)
        {
            gc1 = new GraphConvLayer("gc1", inputDim, hiddenDim, useBias: useBias);
            gc2 = new GraphConvLayer("gc2", hiddenDim, outputDim, useBias: useBias);
            dropout = Dropout(dropoutP);

            RegisterComponents();

            if (device != null && device.type != DeviceType.CPU)
                this.to(device);
        }

        public override Tensor forward(Tensor inputTensor, Tensor adjMat)
        {
            var x = gc1.forward(inputTensor, adjMat);
            x = relu(x);
            x = dropout.forward(x);
            x = gc2.forward(x, adjMat);
            return log_softmax(x, dim: 1);
        }
    }
}
