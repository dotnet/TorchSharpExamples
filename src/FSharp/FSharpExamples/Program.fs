// Learn more about F# at http://docs.microsoft.com/dotnet/fsharp

open System
open System.IO
open System.Reflection

open TorchSharp.Examples
open TorchSharp.Examples.Utils


[<EntryPoint>]
let main args =

    let argumentsPath = Path.Combine(Path.GetDirectoryName(Assembly.GetEntryAssembly().Location), "arguments.json")
    let argumentParser = new ArgumentParser(new FileInfo(argumentsPath), args)

    if argumentParser.Count = 0 then
        argumentParser.UsingMessage("CSharpExamples", "<model-name>")
        1 // return an integer exit code
    else
        
        let epochs = 
            match argumentParser.TryGetValueInt "epochs" with
            | true,e -> e
            | false,_ -> 16

        let timeout = 
            match argumentParser.TryGetValueInt "timeout" with
            | true,t -> t
            | false,_ -> 3600

        for idx = 0 to argumentParser.Count-1 do

            let modelName = argumentParser.[idx]

            match modelName.ToLowerInvariant() with
            | "mnist" -> FSharpExamples.MNIST.run epochs
            | "fgsm"  -> FSharpExamples.AdversarialExampleGeneration.run epochs
            | "alexnet" -> FSharpExamples.AlexNet.run epochs
            | "seq2seq" -> FSharpExamples.SequenceToSequence.run epochs
            | "text" -> FSharpExamples.TextClassification.run epochs
            | _ -> eprintf "Unknown model name"

        0 // return an integer exit code