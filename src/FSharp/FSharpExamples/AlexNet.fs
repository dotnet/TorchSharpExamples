// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
module FSharpExamples.AlexNet

open System
open System.IO
open System.Diagnostics

open TorchSharp
open TorchSharp.Examples

open type TorchSharp.torch.nn
open type TorchSharp.torch.optim
open type TorchSharp.Scalar

// Modified version of original AlexNet to fix CIFAR10 32x32 images.
// 
// The dataset for this example can be found at: https://www.cs.toronto.edu/~kriz/cifar.html
// Download the binary file, and place it in a dedicated folder, e.g. 'CIFAR10,' then edit
// the '_dataLocation' definition below to point at the right folder.
//
// Note: so far, CIFAR10 is supported, but not CIFAR100.

let mutable trainBatchSize = 64
let mutable testBatchSize = 128

let logInterval = 25
let numClasses = 10L

let cmdArgs = Environment.GetCommandLineArgs()
let dataset = "CIFAR10"

let datasetPath = Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "..", "Downloads", dataset)

torch.random.manual_seed(1L) |> ignore

let hasCUDA = TorchText.Datasets.cuda_is_available()

let device = if hasCUDA then torch.CUDA else torch.CPU

let getDataFiles sourceDir targetDir =

    if not (Directory.Exists(targetDir)) then
        Directory.CreateDirectory(targetDir) |> ignore
        Utils.Decompress.ExtractTGZ(Path.Combine(sourceDir, "cifar-10-binary.tar.gz"), targetDir)

type Model(name,device:torch.Device) as this =
    inherit Module<torch.Tensor,torch.Tensor>(name)

    let features = Sequential(("c1", Conv2d(3L, 64L, kernel_size=3L, stride=2L, padding=1L) :> Module<torch.Tensor,torch.Tensor>),
                              ("r1", ReLU(inplace=true) :> Module<torch.Tensor,torch.Tensor>),
                              ("mp1", MaxPool2d(kernel_size=[|2L; 2L|]) :> Module<torch.Tensor,torch.Tensor>),
                              ("c2", Conv2d(64L, 192L, kernel_size=3L, padding=1L) :> Module<torch.Tensor,torch.Tensor>),
                              ("r2", ReLU(inplace=true) :> Module<torch.Tensor,torch.Tensor>),
                              ("mp2", MaxPool2d(kernel_size=[|2L; 2L|]) :> Module<torch.Tensor,torch.Tensor>),
                              ("c3", Conv2d(192L, 384L, kernel_size=3L, padding=1L) :> Module<torch.Tensor,torch.Tensor>),
                              ("r3", ReLU(inplace=true) :> Module<torch.Tensor,torch.Tensor>),
                              ("c4", Conv2d(384L, 256L, kernel_size=3L, padding=1L) :> Module<torch.Tensor,torch.Tensor>),
                              ("r4", ReLU(inplace=true) :> Module<torch.Tensor,torch.Tensor>),
                              ("c5", Conv2d(256L, 256L, kernel_size=3L, padding=1L) :> Module<torch.Tensor,torch.Tensor>),
                              ("r5", ReLU(inplace=true) :> Module<torch.Tensor,torch.Tensor>),
                              ("mp3", MaxPool2d(kernel_size=[|2L; 2L|]) :> Module<torch.Tensor,torch.Tensor>),
                              ("avg", AdaptiveAvgPool2d([|2L; 2L|]) :> Module<torch.Tensor,torch.Tensor>))

    let classifier = Sequential(("d1", Dropout() :> Module<torch.Tensor,torch.Tensor>),
                                ("l1", Linear(256L * 2L * 2L, 4096L) :> Module<torch.Tensor,torch.Tensor>),
                                ("r6", ReLU(inplace=true) :> Module<torch.Tensor,torch.Tensor>),
                                ("d2", Dropout() :> Module<torch.Tensor,torch.Tensor>),
                                ("l2", Linear(4096L, 4096L) :> Module<torch.Tensor,torch.Tensor>),
                                ("r7", ReLU(inplace=true) :> Module<torch.Tensor,torch.Tensor>),
                                ("d3", Dropout() :> Module<torch.Tensor,torch.Tensor>),
                                ("l3", Linear(4096L, numClasses) :> Module<torch.Tensor,torch.Tensor>),
                                ("logsm", LogSoftmax(1L) :> Module<torch.Tensor,torch.Tensor>))

    do
        this.RegisterComponents()

        if device.``type`` = DeviceType.CUDA then
            this.``to``(device) |> ignore

    override _.forward(input) =

        let avg = features.forward(input)
        let x = avg.view([|avg.shape.[0]; 256L*2L*2L|])

        classifier.forward(x)

let loss x y = functional.nll_loss(x,y)

let train (model:Model) (optimizer:Optimizer) (dataLoader: CIFARReader) epoch =

    model.train()

    let size = dataLoader.Size

    let mutable batchID = 1
    let mutable total = 0L
    let mutable correct = 0L

    printfn $"Epoch: {epoch}..."

    for (input,labels) in dataLoader.Data() do

        use d = torch.NewDisposeScope()
        
        optimizer.zero_grad()

        begin
            use estimate = input --> model
            use output = loss estimate labels

            output.backward()
            optimizer.step() |> ignore

            total <- total + labels.shape.[0]

            use sum = estimate.argmax(1L).eq(labels).sum()
            correct <- correct + sum.ToInt64()

            if batchID % logInterval = 0 then
                let count = min (batchID * trainBatchSize) size
                let outputString = output.ToSingle().ToString("0.0000")
                let accString = ((float correct) / (float total)).ToString("0.0000")
                printfn $"\rTrain: epoch {epoch} [{count} / {size}] Loss: {outputString} Acc: {accString}"

            batchID <- batchID + 1
        end

let test (model:Model) (dataLoader:CIFARReader) =
    model.eval()

    let sz = single dataLoader.Size

    let mutable testLoss = 0.0f
    let mutable correct = 0L
    let mutable batchCount = 0L

    for (input,labels) in dataLoader.Data() do

        use d = torch.NewDisposeScope()

        use estimate = input --> model
        use output = loss estimate labels
        testLoss <- testLoss + output.ToSingle()
        batchCount <- batchCount + 1L

        use sum = estimate.argmax(1L).eq(labels).sum()
        correct <- correct + sum.ToInt64()

    let avgLossString = (testLoss / (single batchCount)).ToString("0.0000")
    let accString = ((single correct) / sz).ToString("0.0000")

    printfn $"\rTest set: Average loss {avgLossString} | Accuracy {accString}"
    
    
let trainingLoop (model:Model) epochs trainData testData =
    
        use optimizer = Adam(model.parameters(), 0.001)
        //NN.Optimizer.StepLR(optimizer, 1u, 0.7, last_epoch=5) |> ignore
    
        let sw = Stopwatch()
        sw.Start()
    
        for epoch = 1 to epochs do
            train model optimizer trainData epoch
            test model testData

        sw.Stop()
    
        printfn $"Elapsed time: {sw.Elapsed.TotalSeconds:F1} s."

let run epochs =

    if device.``type`` = DeviceType.CUDA then
        trainBatchSize <- trainBatchSize * 8
        testBatchSize <- testBatchSize * 8

    printfn ""
    printfn $"\tRunning AlexNet with {dataset} on {device.``type``.ToString()} for {epochs} epochs"
    printfn ""

    let targetDir = Path.Combine(datasetPath, "test_data")

    getDataFiles datasetPath targetDir

    use trainData = new CIFARReader(targetDir, false, trainBatchSize, shuffle=true, device=device)
    use testData = new CIFARReader(targetDir, true, testBatchSize, device=device)

    use model = new Model("model", device)

    trainingLoop model epochs trainData testData
    
    ()