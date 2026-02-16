// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System.Diagnostics.CodeAnalysis;
using TorchSharp;

namespace Examples.Utils
{
    public static class TensorExtension
    {
#nullable enable
        public static bool IsNull([NotNullWhen(false)] this torch.Tensor? tensor)
        {
            return tensor is null || tensor.IsInvalid;
        }

        public static bool IsNotNull([NotNullWhen(true)] this torch.Tensor? tensor)
        {
            return !tensor.IsNull();
        }
#nullable disable

        public static T[] ToArray<T>(this torch.Tensor tensor) where T : unmanaged
        {
            return tensor.cpu().data<T>().ToArray();
        }

        public static T ToItem<T>(this torch.Tensor tensor) where T : unmanaged
        {
            return tensor.cpu().item<T>();
        }
    }
}