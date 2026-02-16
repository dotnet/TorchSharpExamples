// Copyright (c) .NET Foundation and Contributors.  All Rights Reserved.  See LICENSE in the project root for license information.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Examples.Utils
{
    public class RobertaTokenizer
    {
        private const string EncoderJsonName = "encoder.json";
        private const string MergeName = "vocab.bpe";
        private const string DictName = "dict.txt";

        private static readonly Uri EncoderJsonUrl = new("https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json");
        private static readonly Uri MergeUrl = new("https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe");
        private static readonly Uri DictUrl = new("https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt");

        private readonly string _path;
        private readonly RobertaVocabulary Vocabulary;
        private readonly IReadOnlyDictionary<string, int> Encoder;
        private readonly IReadOnlyDictionary<int, string> Decoder;
        private readonly (string, string)[] Merges;
        private readonly DefaultDictionary<(string, string), int> MergeRanks;
        public readonly IReadOnlyDictionary<char, char> ByteToUnicode;
        private readonly IReadOnlyDictionary<char, char> UnicodeToByte;
        //private const string Pattern = @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
        public readonly char StartChar;

        public int VocabSize => Vocabulary.Count;
        public int PadIndex => Vocabulary.PadIndex;
        public int UnkIndex => Vocabulary.UnkIndex;
        public int BosIndex => Vocabulary.BosIndex;
        public int EosIndex => Vocabulary.EosIndex;
        public int MaskIndex => Vocabulary.MaskIndex;

        public string PadToken => Vocabulary.PadWord;
        public string UnkToken => Vocabulary.UnkWord;
        public string BosToken => Vocabulary.BosWord;
        public string EosToken => Vocabulary.EosWord;
        public string MaskToken => Vocabulary.MaskWord;

        public RobertaTokenizer(string path, char startChar = '\u0120')
        {
            _path = path;
            Directory.CreateDirectory(_path);

            StartChar = startChar;
            Vocabulary = GetVocabulary();
            Encoder = GetEncoder();
            Decoder = Encoder.Reverse();
            Merges = GetMerges();
            MergeRanks = GetMergeRanks();
            ByteToUnicode = GetByteToUnicode();
            UnicodeToByte = ByteToUnicode.Reverse();
        }

        #region Public API
        /// <summary>
        /// Tokenize a sequence of words into a sequence of subtokens.
        /// </summary>
        public IList<string> Tokenize(IEnumerable<string> words)
        {
            return words
                .Select(word => Bpe(word))
                .Aggregate((all, tokens) => all.Concat(tokens))
                .ToArray();
        }

        /// <summary>
        /// Tokenize a sentence into a sequence of subtokens.
        /// The sentence will first be split into a sequence of words according to the given Regex pattern
        /// before applying the tokenization algorithm.
        /// </summary>
        public IList<string> Tokenize(string sentence, string splitPattern = @"\s+")
        {
            sentence = Normalize(sentence);
            var words = Regex.Split(sentence, splitPattern).Select(word => StartChar + word).ToArray();
            //words[0] = words[0][1..];   // Do not add the StartChar to the first word
            return Tokenize(words);
        }

        /// <summary>
        /// Tokenize a sequence of words into a sequence of subtoken IDs.
        /// </summary>
        public IList<int> TokenizeToId(IEnumerable<string> words)
        {
            return words
                .Select(word => TokensToIds(Bpe(word)) as IEnumerable<int>)
                .Aggregate((all, tokens) => all.Concat(tokens))
                .ToArray();
        }

        /// <summary>
        /// Tokenize a sentence into a sequence of subtoken IDs.
        /// The sentence will first be split into a sequence of words according to the given Regex pattern
        /// before applying the tokenization algorithm.
        /// </summary>
        public IList<int> TokenizeToId(string sentence, string splitPattern = @"\s+")
        {
            sentence = Normalize(sentence);
            var words = Regex.Split(sentence, splitPattern).Select(word => StartChar + word).ToArray();
            //words[0] = words[0][1..];   // Do not add the StartChar to the first word
            return TokenizeToId(words);
        }

        /// <summary>
        /// Convert a sequence of subtokens into their IDs.
        /// </summary>
        public IList<int> TokensToIds(IEnumerable<string> tokens)
        {
            return tokens.Select(token =>
            {
                var id = Encoder[token];
                return id <= 0 ? -id : Vocabulary.IndexOf($"{id}");
            })
            .ToArray();
        }

        /// <summary>
        /// Untokenize a sequence of subtoken IDs into the corresponding sentence text.
        /// </summary>
        public string Untokenize(IEnumerable<int> tokenIds)
        {
            return DecodeFromConverted(tokenIds);
        }

        /// <summary>
        /// Untokenize a sequence of subtoken IDs into the corresponding tokens text.
        /// </summary>
        public IList<string> UntokenizeToTokens(IEnumerable<int> tokenIds)
        {
            return DecodeConvertedToTokens(tokenIds).ToArray();
        }
        #endregion

        #region Normalization
        /// <summary>
        /// Modify some symbols to normalize a string text.
        /// </summary>
        private static string Normalize(string input)
        {
            var processed = input
                .Trim()
                .Replace("``", "\"")
                .Replace("''", "\"")
                .Replace("\\\"", "\"");
            return processed;
        }

        #endregion

        #region Initialization
        private RobertaVocabulary GetVocabulary()
        {
            _ = LoadFromFileOrDownloadFromWeb(_path, DictName, DictUrl);
            return new RobertaVocabulary(Path.Join(_path, DictName));
        }

        private Dictionary<string, int> GetEncoder()
        {
            var contents = LoadFromFileOrDownloadFromWeb(_path, EncoderJsonName, EncoderJsonUrl);

            // Parse JSON
            try
            {
                var jsonResult = (Dictionary<string, int>)JsonSerializer.Deserialize(
                    contents, typeof(Dictionary<string, int>));

                jsonResult[Vocabulary.BosWord] = -Vocabulary.BosIndex;
                jsonResult[Vocabulary.EosWord] = -Vocabulary.EosIndex;
                jsonResult[Vocabulary.UnkWord] = -Vocabulary.UnkIndex;
                jsonResult[Vocabulary.PadWord] = -Vocabulary.PadIndex;

                return jsonResult;
            }
            catch (JsonException e)
            {
                throw new JsonException($"Problems met when parsing JSON object in {EncoderJsonName}.\n" +
                                        $"Error message: {e.Message}");
            }
        }

        private (string, string)[] GetMerges()
        {
            var contents = LoadFromFileOrDownloadFromWeb(_path, MergeName, MergeUrl);

            // Parse merge info
            try
            {
                var merges = contents.Split('\n')[1..^1].Select(line =>
                {
                    var split = line.Split(' ');
                    if (split[0] == "" || split[1] == "")
                    {
                        throw new Exception("Invalid format of merge file: \"{line}\"");
                    }
                    return (split[0], split[1]);
                }).ToArray();
                return merges;
            }
            catch (Exception e)
            {
                throw new Exception($"Problems met when parsing records in {MergeName}.\n" +
                                    $"Error message: {e.Message}");
            }
        }

        private DefaultDictionary<(string, string), int> GetMergeRanks()
        {
            var mergeRanks = new DefaultDictionary<(string, string), int>(() => int.MaxValue);
            for (var i = 0; i < Merges.Length; ++i)
            {
                mergeRanks.Add(Merges[i], i);
            }

            return mergeRanks;
        }

        /// <summary>
        /// Returns list of utf-8 bytes and a corresponding list of unicode chars.
        /// This mapping is to make unseen characters (such as control characters) displayable.
        /// </summary>
        private static Dictionary<char, char> GetByteToUnicode()
        {
            var byteToUnicode = Enumerable.Range('!', '~' - '!' + 1)
                .Concat(Enumerable.Range('¡', '¬' - '¡' + 1))
                .Concat(Enumerable.Range('®', 'ÿ' - '®' + 1))
                .ToDictionary(b => (char)b, b => (char)b);

            const int numChars = 256;
            var n = 0;
            foreach (var b in Enumerable.Range(0, numChars))
            {
                if (byteToUnicode.ContainsKey((char)b)) continue;
                byteToUnicode.Add((char)b, (char)(numChars + n));
                ++n;
            }

            byteToUnicode.Add('Ġ', 'Ġ');    // Space char

            return byteToUnicode;
        }

        public static string LoadFromFileOrDownloadFromWeb(string path, string fileName, Uri url)
        {
            var contents = string.Empty;
            var filePath = Path.Join(path, fileName);
            if (!File.Exists(filePath))
            {
                try
                {
                    using var webClient = new WebClient();
                    contents = webClient.DownloadString(url);
                }
                catch (WebException e)
                {
                    throw new WebException($"File {fileName} not found and cannot be downloaded from {url}.\n" +
                                           $"Error message: {e.Message}");
                }

                try
                {
                    File.WriteAllText(filePath, contents);
                    Console.WriteLine($"File {fileName} successfully downloaded from {url} and saved to {path}.");
                }
                catch (Exception e)
                {
                    Console.WriteLine($"{DateTime.Now} - WARNING: File {fileName} successfully downloaded from {url}, " +
                                      $"but error occurs when saving file {fileName} into {path}.\n" +
                                      $"Error message: {e.Message}");
                }

            }
            else
            {
                try
                {
                    contents = File.ReadAllText(filePath);
                }
                catch (Exception e)
                {
                    throw new IOException($"Problems met when reading {filePath}.\n" +
                                          $"Error message: {e.Message}");
                }
            }

            return contents;
        }
        #endregion

        #region Wrapper for BPE algorithm
        /// <summary>
        /// Decode converted token IDs and return the corresponding string.
        /// Origin token ID is the token ID after BPE processing, and <see cref="Vocabulary"/> defines a mapping
        ///     between origin token IDs and converted token IDs.
        /// </summary>
        private string DecodeFromConverted(IEnumerable<int> tokenIds)
        {
            var tokens = DecodeConvertedToTokens(tokenIds);
            var text = string.Join(string.Empty, tokens).Replace(StartChar, ' ').Trim();
            return text;
        }

        /// <summary>
        /// Decode converted token IDs and return the corresponding tokens.
        /// Origin token ID is the token ID after BPE processing, and <see cref="Vocabulary"/> defines a mapping
        ///     between origin token IDs and converted token IDs.
        /// </summary>
        private IEnumerable<string> DecodeConvertedToTokens(IEnumerable<int> tokenIds)
        {
            // 1. not to decode padding tokens
            // 2. special tokens (BOS, EOS, PAD, UNK) in vocabulary are presented as strings rather than integers,
            //    so they will cause parsing failure. We treat their IDs as negative integers to avoid conflict with
            //    normal tokens. Other unrecognized tokens will be treated as token#13, which is ".".
            var tokenArray = tokenIds
                .Where(token => token != Vocabulary.PadIndex)
                .Select(token => int.TryParse(Vocabulary[token], out var result) ? result :
                    token < RobertaVocabulary.NumSpecialSymbols ? -token :
                    13)
                .ToArray();
            var tokens = tokenArray.Select(id =>
                new string(Decoder[id]
                    .Where(c => UnicodeToByte.ContainsKey(c))
                    .Select(c => UnicodeToByte[c]).ToArray()));
            return tokens;
        }
        #endregion

        #region BPE algorithm
        /// <summary>
        /// Encode text with several tokens into BPE-ed sub-tokens.
        /// </summary>
        private IEnumerable<string> Bpe(string word)
        {
            var convertedToken = string.Join("", word
                .Where(ByteToUnicode.ContainsKey)
                .Select(b => ByteToUnicode[b]));
            if (convertedToken.Length == 0) return Array.Empty<string>();
            return BpeToken(convertedToken);
        }

        /// <summary>
        /// Encode a token into BPE-ed sub-tokens. E.g., "playing" into ["play", "ing"].
        /// </summary>
        private IEnumerable<string> BpeToken(string token)
        {
            var word = token.Select(c => c.ToString()).ToList();
            var pairs = WordToPairs(word);

            if (pairs.Count == 0)
            {
                return new List<string> { token };
            }

            while (true)
            {
                /* while conditions */
                // if only one element left, merge is finished (with the whole word merged)
                if (word.Count == 1)
                {
                    break;
                }

                // get the most frequent bi-gram pair
                var (first, second) = pairs.ArgMin(pair => MergeRanks[pair]);
                if (!MergeRanks.ContainsKey((first, second)))
                {
                    break;
                }
                /* end while conditions */

                // search and merge all (first, second) pairs in {word}
                var newWord = new List<string>();
                var i = 0;
                while (i < word.Count)
                {
                    // find the next occurrence of {first} and add the elements before into {newWord}
                    var j = word.IndexOf(first, i);
                    if (j == -1)
                    {
                        newWord.AddRange(word.Skip(i));
                        break;
                    }
                    else
                    {
                        newWord.AddRange(word.Skip(i).Take(j - i));
                        i = j;
                    }

                    // check the next element is {second} or not
                    if (i < word.Count - 1 && word[i + 1] == second)
                    {
                        newWord.Add(first + second);
                        i += 2;
                    }
                    else
                    {
                        newWord.Add(word[i]);
                        i += 1;
                    }
                }

                word = newWord;

                // otherwise, continue merging
                pairs = WordToPairs(word);
            }

            return word;
        }

        /// <summary>
        /// Extract element pairs in an aggregating word. E.g. [p, l, ay] into [(p,l), (l,ay)].
        /// If word contains 0 or 1 element, an empty HashSet will be returned.
        /// </summary>
        private static HashSet<(string, string)> WordToPairs(IReadOnlyList<string> word)
        {
            var pairs = new HashSet<(string, string)>();
            if (word.Count <= 1) return pairs;

            var prevElem = word[0];
            foreach (var elem in word.Skip(1))
            {
                pairs.Add((prevElem, elem));
                prevElem = elem;
            }

            return pairs;
        }
        #endregion
    }

    internal static class IEnumerableExtension
    {
        public static T ArgMin<T>(this IEnumerable<T> source, Func<T, int> getValue)
        {
            var keys = source.ToList();     // avoid enumerate twice
            var values = keys.Select(getValue);
            var (minSource, minValue) = keys.Zip(values).Aggregate((min, x) => min.Second <= x.Second ? min : x);
            return minValue < int.MaxValue ? minSource : default;
        }
    }

    internal static class IReadOnlyDictionaryExtension
    {
        public static IReadOnlyDictionary<TValue, TKey> Reverse<TKey, TValue>(this IReadOnlyDictionary<TKey, TValue> source)
        {
            var dictionary = new Dictionary<TValue, TKey>();
            foreach (var (key, value) in source)
            {
                dictionary[value] = key;
            }
            return dictionary;
        }
    }

}