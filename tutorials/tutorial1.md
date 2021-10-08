# Setting Things Up

To use TorchSharp, you need some packages from NuGet.

First and foremost, you need to download the most recent version of the `TorchSharp` packaage at [https://www.nuget.org/packages/TorchSharp/](https://www.nuget.org/packages/TorchSharp/). That's the .NET bindings to libtorch, and it contains the .NET API.

However, you also need one of several packages containing distributions of libtorch itself, the highly capable native-code engine behind PyTorch. Because there are several configurations of it, TorchSharp does not contain a reference to the backend package, you need to pick one and add to your project(s).

The basic backend supports training and inference on CPUs, but there is also support for CUDA on Windows and Linux, for use on machines with compatible hardware. Using CUDA for training can speed things up by orders of magnitude, so it's important to use the right backend.

These are the various libtorch packages:

|Name|URL|Description|
|-----|-----------------|--------------|
|libtorch-cpu|https://www.nuget.org/packages/libtorch-cpu/|A CPU backend, which works on Windows, Linus, and MacOS|
|libtorch-cpu-win-x64|https://www.nuget.org/packages/libtorch-cpu-win-x64/|A CPU backend with only Windows binaries|
|libtorch-cpu-linux-x64|https://www.nuget.org/packages/libtorch-cpu-linux-x64/|A CPU backend with only Linux binaries|
|libtorch-cpu-osx-x64|https://www.nuget.org/packages/libtorch-cpu-osx-x64/|A CPU backend with only OSX binaries|
|libtorch-cuda-11.1-win-x64|https://www.nuget.org/packages/libtorch-cpu-osx-x64https://www.nuget.org/packages/libtorch-cuda-11.1-win-x64//|A CUDA backend for Windows and CUDA 11.1|
|libtorch-cuda-11.1-linux-x64|https://www.nuget.org/packages/libtorch-cpu-osx-x64https://www.nuget.org/packages/libtorch-cuda-11.1-linux-x64//|A CUDA backend for Ubuntu Linux and CUDA 11.1|

# Usings

Once you have the right NuGet packages, the next thing is to get the right usings directives at the top of your source files. TorchSharp consists of a lot of namespaces and static classes, and to make programming TorchSharp convenient, you usually need to include a several of them.

A typical set of usings looks like this:

```C#
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
```

However, for these tutorials, it would obscure the API to have too many usings. It's better, for pedagocial reasons, to explicitly qualify names until their scope becomes well known. So, the tutorials will generally either use only the minimal `using TorchSharp`, or explicitly list a longer list of usings.

