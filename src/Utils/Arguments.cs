using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Newtonsoft.Json;

namespace TorchSharp.Examples.Utils
{
    [JsonObject]
    public sealed class ArgumentDescriptor
    {
        /// <summary>
        /// Long names are used with '--' and can be any one word using letters and numbers.
        /// The long name spelling are not case-sensitive.
        /// </summary>
        [JsonProperty(Required = Required.Always)]
        public string LongName { get; set; }

        /// <summary>
        /// Short names must be a single character, and are sensitive to case.
        /// </summary>
        [JsonProperty(Required = Required.Default)]
        public string ShortName { get; set; }

        /// <summary>
        /// If true, the parser should allow multiple values.
        /// </summary>
        [JsonProperty(Required = Required.Default)]
        public bool AllowMultiple { get; set; }

        /// <summary>
        /// The kind of argument.
        /// </summary>
        [JsonProperty(Required = Required.Always)]
        public ArgumentType ArgType { get; set; }

        /// <summary>
        /// An explanation of the argument, intended for human consumption as part of a 'using' message.
        /// </summary>
        public String Explanation { get; set; }

        public enum ArgumentType
        {
            /// <summary>
            /// A string argument.
            /// </summary>
            /// <example>
            /// --name=foobar
            /// </example>
            String,
            /// <summary>
            /// An integer argument.
            /// </summary>
            /// <example>
            /// --count=10
            /// </example>
            Integer,
            /// <summary>
            /// An comma-separated list of strings.
            /// </summary>
            /// <example>
            /// --options=a,b,c
            /// </example>
            List,
            /// <summary>
            /// A boolean argument, for example
            /// </summary>
            /// <example>
            /// --doit=true
            /// </example>
            Boolean,
            /// <summary>
            /// A Boolean flag that requires no value. Absence is 'false'
            /// </summary>
            /// <example>
            /// --doit
            /// </example>
            Flag,
        }
    }
}
