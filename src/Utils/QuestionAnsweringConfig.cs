// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.

namespace Examples.Utils
{
    public class QuestionAnsweringConfig
    {
        public string LoadModelPath { get; set; }
        public string DataDir { get; set; }
        public string TrainFile { get; set; }
        public string ValidFile { get; set; }
        public string TestFile { get; set; }
        public string VocabDir { get; set; }

        public int BatchSize { get; set; }
        public int OptimizeSteps { get; set; }
        public int MaxSequence { get; set; }
        public bool Cuda { get; set; }
        public string SaveDir { get; set; }

        public double LearningRate { get; set; }
        public int LogEveryNSteps { get; set; }
        public int ValidateEveryNSteps { get; set; }

        public int TopK { get; set; }
    }

}