using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Threading.Tasks;

using Newtonsoft.Json;

namespace TorchSharp.Examples.Utils
{
    /// <summary>
    /// Yet another argument parser.
    /// </summary>
    public sealed class ArgumentParser
    {
        public ArgumentParser(FileInfo argumentConfiguration, IList<string> args)
        {
            Initialize(File.ReadAllText(argumentConfiguration.FullName), args);
        }

        public ArgumentParser(string argumentConfiguration, IList<string> args)
        {
            Initialize(argumentConfiguration, args);
        }

        public void UsingMessage(string name, string positionals)
        {
            Console.Error.WriteLine("using:");
            Console.Error.Write($"{name} ");
            foreach (var desc in descriptors)
            {
                Console.Error.Write($"[--{desc.LongName} | -{desc.ShortName}] {desc.ArgType.ToString().ToLower()} ");
            }

            Console.Error.WriteLine($"{positionals}...");

            foreach (var desc in descriptors)
            {
                Console.Error.WriteLine($"--{desc.LongName} | -{desc.ShortName}: {desc.ArgType.ToString().ToLower()}, {desc.Explanation} ");
            }
        }
        public int Count => positionalArguments.Count;

        public string this[int index]
        {
            get { return positionalArguments[index]; }
        }

        private void Initialize(string argumentConfiguration, IList<string> args)
        {
            try
            {
                descriptors = JsonConvert.DeserializeObject<List<ArgumentDescriptor>>(argumentConfiguration);

                for (int idx = 0; idx < args.Count; ++idx)
                {
                    var arg = args[idx];

                    if (arg.StartsWith("--"))
                    {
                        // Long form argument, --name=value, --name:value, or --name value
                        string[] kv = null;

                        if (arg.Contains(':'))
                        {
                            kv = arg.Substring(2).Split(':');
                        }
                        else if (arg.Contains('='))
                        {
                            kv = arg.Substring(2).Split('=');
                        }
                        else
                        {
                            kv = new string[] { arg.Substring(2) };
                        }

                        ProcessArgument(kv, args, descriptors, false, ref idx);
                    }
                    else if (arg.StartsWith("-"))
                    {
                        // Short form argument, -v value
                        var key = arg.Substring(1);

                        if (key.Length == 1)
                        {
                            ProcessArgument(new string[] { key }, args, descriptors, true, ref idx);
                        }
                        else
                        {
                            ProcessFlags(key, args, descriptors);
                        }
                    }
                    else
                    {
                        // Positional argument, always interpreted as a string
                        positionalArguments.Add(arg);
                    }
                }
            }
            catch (Exception e)
            {
                Console.Error.WriteLine($"Internal error reading command arguments definition file: {e.Message}");
            }
        }

        private void ProcessFlags(string key, IList<string> args, List<ArgumentDescriptor> arguments)
        {
            foreach (var ch in key)
            {
                var name = ch.ToString();

                foreach (var argDesc in arguments)
                {
                    if (name.Equals(argDesc.ShortName))
                    {
                        if (argDesc.ArgType != ArgumentDescriptor.ArgumentType.Flag)
                        {
                            Console.Error.WriteLine("Mulitple short-form arguments are only valid if they do not take a value.");
                            continue;
                        }
                        namedArguments.Add(argDesc.LongName, true);
                        break;
                    }
                }
            }
        }

        private void ProcessArgument(string[] kv, IList<string> args, List<ArgumentDescriptor> arguments, bool shortForm, ref int idx)
        {
            var name = kv[0];

            var argType = ArgumentDescriptor.ArgumentType.Flag;

            foreach (var argDesc in arguments)
            {
                if (!shortForm && name.ToLowerInvariant().Equals(argDesc.LongName.ToLowerInvariant()) ||
                    shortForm && name.Equals(argDesc.ShortName))
                {
                    argType = argDesc.ArgType;
                    name = argDesc.LongName;
                    break;
                }
            }

            try
            {
                switch (argType)
                {
                    case ArgumentDescriptor.ArgumentType.Flag:
                        namedArguments.Add(name, true);
                        break;
                    case ArgumentDescriptor.ArgumentType.Boolean:
                        {
                            if (bool.TryParse((kv.Length == 1) ? args[++idx] : kv[1], out bool value))
                            {
                                namedArguments.Add(name, value);
                            }
                            break;
                        }
                    case ArgumentDescriptor.ArgumentType.Integer:
                        {
                            if (int.TryParse((kv.Length == 1) ? args[++idx] : kv[1], out int value))
                            {
                                namedArguments.Add(name, value);
                            }
                            break;
                        }
                    case ArgumentDescriptor.ArgumentType.String:
                        {
                            var value = (kv.Length == 1) ? args[++idx] : kv[1];
                            namedArguments.Add(name, value);
                            break;
                        }
                    case ArgumentDescriptor.ArgumentType.List:
                        {
                            var value = ((kv.Length == 1) ? args[++idx] : kv[1]).Split(',');
                            namedArguments.Add(name, value);
                            break;
                        }
                }
            }
            catch(ArgumentOutOfRangeException)
            {
            }
        }

        public bool TryGetValueBool(string name, out bool value)
        {
            return TryGetValue<bool>(name, out value);
        }

        public bool TryGetValueInt(string name, out int value)
        {
            return TryGetValue<int>(name, out value);
        }

        public bool TryGetValueString(string name, out string value)
        {
            return TryGetValue<string>(name, out value);
        }

        public bool TryGetValueStrings(string name, out string[] value)
        {
            return TryGetValue<string[]>(name, out value);
        }

        public bool TryGetValue<T>(string name, out T value, T @default = default(T))
        {
            if (namedArguments.TryGetValue(name, out var obj) && obj is T)
            {
                value = (T)obj;
                return true;
            }
            value = @default;
            return false;
        }

        private List<ArgumentDescriptor> descriptors = null;

        private Dictionary<string, object> namedArguments = new Dictionary<string, object>();
        private List<string> positionalArguments = new List<string>();
    }
}
