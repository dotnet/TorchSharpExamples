// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Examples.Utils
{
    /// <summary>
    /// A mapping from symbols to consecutive integers.
    /// </summary>
    public class RobertaVocabulary
    {
        public const int NumSpecialSymbols = 4;

        public string PadWord { get; }
        public string EosWord { get; }
        public string UnkWord { get; }
        public string BosWord { get; }

        public int PadIndex { get; }
        public int EosIndex { get; }
        public int UnkIndex { get; }
        public int BosIndex { get; }

        public string MaskWord { get; private set; }
        public int MaskIndex { get; private set; }

        private readonly List<string> _symbols;
        private readonly Dictionary<string, int> _indices;
        private readonly List<int> _counter;

        public IReadOnlyDictionary<string, int> Indices => _indices;

        /// <summary>
        /// Loads the vocabulary from a text file with the format:
        ///     <symbol0> <count0>
        ///     <symbol1> <count1>
        ///     ...
        /// </summary>
        /// <exception cref="ArgumentNullException">Any of `pad`, `eos`, `unk` and `bos` is `null`.</exception>
        public RobertaVocabulary(string fileName, string pad = "<pad>", string eos = "</s>", string unk = "<unk>", string bos = "<s>",
            string[] extraSpecialSymbols = null)
        {
            _indices = new Dictionary<string, int>();
            _counter = new List<int>();
            _symbols = new List<string>();

            PadWord = pad;
            EosWord = eos;
            UnkWord = unk;
            BosWord = bos;
            BosIndex = AddSymbol(bos);
            PadIndex = AddSymbol(pad);
            EosIndex = AddSymbol(eos);
            UnkIndex = AddSymbol(unk);

            if (extraSpecialSymbols != null)
            {
                foreach (var symbol in extraSpecialSymbols)
                {
                    AddSymbol(symbol);
                }
            }

            AddFromFile(fileName);
        }

        /// <summary>
        /// Add a word to the vocabulary.
        /// </summary>
        /// <exception cref="ArgumentNullException">`word` is `null`.</exception>
        private int AddSymbol(string word, int n = 1)
        {
            if (word == null)
            {
                throw new ArgumentNullException(nameof(word), $"argument {nameof(word)} should not be null.");
            }

            int idx;
            if (_indices.ContainsKey(word))
            {
                idx = _indices[word];
                _counter[idx] += n;
            }
            else
            {
                idx = _symbols.Count;
                _indices[word] = idx;
                _symbols.Add(word);
                _counter.Add(n);
            }

            return idx;
        }

        public int AddMaskSymbol(string mask = "<mask>")
        {
            MaskWord = mask;
            MaskIndex = AddSymbol(mask);
            return MaskIndex;
        }

        /// <exception cref="ArgumentOutOfRangeException">`idx` is negative.</exception>
        public string this[int idx]
        {
            get
            {
                if (idx < 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(idx), $"Index should be non-negative, got {idx}.");
                }

                return idx < _symbols.Count ? _symbols[idx] : UnkWord;
            }
        }

        public int Count => _symbols.Count;

        public bool Contains(string symbol) => symbol != null && _indices.ContainsKey(symbol);

        /// <exception cref="ArgumentNullException">`symbol` is `null`.</exception>
        public int IndexOf(string symbol) => _indices.ContainsKey(symbol) ? _indices[symbol] : UnkIndex;

        /// <summary>
        /// Loads a pre-existing vocabulary from a text file and adds its symbols to this instance.
        /// </summary>
        private void AddFromFile(string fileName)
        {
            var lines = File.ReadAllLines(fileName, Encoding.UTF8);

            foreach (var line in lines)
            {
                var splitLine = line.Trim().Split(' ');
                if (splitLine.Length != 2)
                {
                    throw new FileFormatException("Incorrect vocabulary format, expected \"<token> <cnt>\"");
                }

                var word = splitLine[0];
                if (int.TryParse(splitLine[1], out var count))
                {
                    AddSymbol(word, count);
                }
                else
                {
                    throw new FileFormatException($"Cannot parse {splitLine[1]} as an integer. File line: \"{line}\".");
                }
            }
        }
    }

    public class FileFormatException : Exception
    {
        public FileFormatException()
        {
        }

        public FileFormatException(string message) : base(message)
        {
        }

        public FileFormatException(string message, Exception innerException) : base(message, innerException)
        {
        }
    }

}