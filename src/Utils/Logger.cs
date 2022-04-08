// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;

namespace Examples.Utils
{
    public class Logger<T>
    {
        private readonly Type Type = typeof(T);

        public void Log(string text = "", bool newline = true, bool carriageReturn = false, LogLevel logLevel = LogLevel.Info)
        {
            Console.Write(
                $"{(carriageReturn ? "\r" : "")}" +
                $"{DateTime.Now:u} - {Type.Name} {logLevel}: {text}" +
                $"{(newline ? "\n" : "")}");
        }

        public void LogLoop(string text = "")
        {
            Console.Write($"\r{Type.Name}: {text}");
        }

        public void LogAppend(string text = "")
        {
            Console.WriteLine(text);
        }
    }

    public enum LogLevel
    {
        Info,
        Debug,
        Warning,
        Error
    }
}