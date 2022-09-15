[![Gitter](https://badges.gitter.im/dotnet/TorchSharp.svg)](https://gitter.im/dotnet/TorchSharp?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
<br/>
# TorchSharp Examples

This repo holds examples and tutorials related to [TorchSharp](https://github.com/dotnet/TorchSharp), .NET-only bindings to libtorch, the engine behind PyTorch. If you are trying to familiarize yourself with TorchSharp, rather than contributing to it, this is the place to go.

Currently, the examples are the same that are also found in the TorchSharp repo. Unlike the setup in that repo, where the examples are part of the overall VS solution file and use project references to pick up the TorchSharp dependencies, in this repo, the example solution is using the publically available TorchSharp packages form NuGet.

The examples and tutorials assume that you are on the latest version of TorchSharp, which currently is 0.97.5.

### System / Environment Requirements

In order to use TorchSharp, you will need both the most recent TorchSharp package, as well as one of the several libtorch-* packages that are available. The most basic one, which is used in this repository, is the libtorch-cpu package. As the name suggests, it uses a CPU backend to do training and inference.

There is also support for CUDA 11.3 on both Windows and Linux, and each of these combinations has its own NuGet package. If you want to train on CUDA, you need to replace references to libtorch-cpu in the solution and projects.

__Note__: Starting with NuGet release 0.93.4, we have simplified the package structure, so you only need to select one of these three packages, and it will include the others:

    TorchSharp-cpu
    TorchSharp-cuda-windows
    TorchSharp-cuda-linux

The examples solution should build without any modifications, either with Visual Studio, or using `dotnet build'. All of the examples build on an Nvidia GPU with 8GB of memory, while only a subset build on a GPU with 6GB. Running more than a few epochs while training on a CPU will take a very long time, especially on the CIFAR10 examples. MNIST is the most reasonable example to train on a CPU.

## Structure

There are variants of all models in both C# and F#. For C#, there is a 'Models' library, and a 'XXXExamples' console app, which is what is used for batch training of the model. For F#, the models are bundled with the training code (we may restructure this in the future). There is also a utility library that is written in C# only, and used from both C# and F#.

The console apps are, as mentioned, meant to be used for batch training. The command line must specify the model to be used. In the case of MNIST, there are two data sets -- the original 'MNIST' as well as the harder 'Fashion MNIST'.

The repo contains no actual data sets. You have to download them manually and, in some cases, extract the data from archives.

## Data Sets

The MNIST model uses either:

* [MNIST](http://yann.lecun.com/exdb/mnist/)
    
* [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion)

Both sets are 28x28 grayscale images, archived in .gz files.

The AlexNet, ResNet*, MobileNet, and VGG* models use the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) data set. Instructions on how to download it is available in the CIFAR10 source files.

SequenceToSequence uses the [WikiText2](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip) dataset. It's kept in a regular .zip file.

TextClassification uses the [AG_NEWS](https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv) dataset, a CSV file.

# Tutorials

We have started work on tutorials, but they are not ready yet. They will mostly be based on .NET Interactive notebooks. If you haven't tried that environment yet, it's worth playing around with it inside VS Code.

# Contributing

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

There are two main things we would like help with:

1. Adding completely new examples. File an issue and assign it to yourself, so we can track it.

2. Picking up an issue from the 'Issues' list. For example, the examples are currently set up to run on Windows, picking up data from under the 'Downloads' folder. If you have thoughts on the best way to do this on MacOS or Linux, please help with that.

If you add a new example, please adjust it to work on a mainstream CUDA processor. This means making sure that it builds on an 8GB processor, with sufficient invocations of the garbage collector.

## A Useful Tip for Contributors

A useful tip from the Tensorflow.NET repo:

After you fork, add dotnet/TorchSharp as 'upstream' to your local repo ...

```git
git remote add upstream https://github.com/dotnet/TorchSharpExamples.git
```

This makes it easy to keep your fork up to date by regularly pulling and merging from upstream.

Assuming that you do all your development off your main branch, keep your main updated
with these commands:

```git
git checkout main
git pull upstream main
git push origin main
```

Then, you merge onto your dev branch:

```git
git checkout <<your dev branch>>
git merge main
```
