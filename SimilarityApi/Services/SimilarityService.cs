using FastText.NetWrapper;
using SimilarityApi.Models;
using System;
using System.Net.Http;
using System.Text.Json;
using System.Text;
using System.Text.RegularExpressions;

namespace SimilarityApi.Services;
public interface ISimilarityService
{
    Task<Similarity> CalculateSimilarity(string sentence1, string sentence2);
}
public class SimilarityService : ISimilarityService
{
    private readonly FastTextWrapper _fastTextModel;
    private static HttpClient _httpClient;
    public SimilarityService()
    {
        _httpClient = new HttpClient();
        // Load pre-trained FastText model (replace with actual path to your model)
        _fastTextModel = new FastTextWrapper();
        //_fastTextModel.LoadModel("path/to/your/fasttext/model.bin");
    }
    public async Task<Similarity> CalculateSimilarity(string sentence1, string sentence2)
    {
        // Simple similarity calculation (e.g., Jaccard index, Levenshtein distance, etc.)
        var Similarities = new Similarity
        {
            LevenshteinDistance = CalculateLevenshteinDistance(sentence1, sentence2),
            Jaccard = CalculateJaccard(sentence1, sentence2),
            FastText = 0,//CalculateFastText(sentence1, sentence2)
            TfidfCosine = CalculateTfidfCosineSimilarity(sentence1, sentence2),
            SentenceTransformer = await CalculateSentenceTransformerSimilarity(sentence1, sentence2)
        };
        return Similarities;
    }
    #region Levenshtein Distance
    private double CalculateLevenshteinDistance(string sentence1, string sentence2)
    {
        int distance = LevenshteinDistance(sentence1, sentence2);
        int maxLen = Math.Max(sentence1.Length, sentence2.Length);
        return 1.0 - (double)distance / maxLen;
    }
    private int LevenshteinDistance(string s1, string s2)
    {
        int[,] d = new int[s1.Length + 1, s2.Length + 1];

        for (int i = 0; i <= s1.Length; i++)
            d[i, 0] = i;
        for (int j = 0; j <= s2.Length; j++)
            d[0, j] = j;

        for (int i = 1; i <= s1.Length; i++)
        {
            for (int j = 1; j <= s2.Length; j++)
            {
                int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
                d[i, j] = Math.Min(
                    Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1),
                    d[i - 1, j - 1] + cost);
            }
        }

        return d[s1.Length, s2.Length];
    }
    #endregion
    #region Jaccard
    private double CalculateJaccard(string sentence1, string sentence2)
    {
        var set1 = new HashSet<string>(sentence1.Split(' '));
        var set2 = new HashSet<string>(sentence2.Split(' '));

        var intersection = new HashSet<string>(set1);
        intersection.IntersectWith(set2);

        var union = new HashSet<string>(set1);
        union.UnionWith(set2);

        return intersection.Count == 0 ? 0.0 : (double)intersection.Count / union.Count;
    }
    #endregion
    #region FastText
    private double CalculateFastText(string sentence1, string sentence2)
    {
        var vector1 = GetSentenceVector(sentence1);
        var vector2 = GetSentenceVector(sentence2);

        return CosineSimilarity(vector1, vector2);
    }
    private float[] GetSentenceVector(string sentence)
    {
        var words = sentence.Split(' ');
        var vectors = words.Select(word => _fastTextModel.GetWordVector(word)).ToArray();
        var sentenceVector = new float[300];

        foreach (var vector in vectors)
        {
            for (int i = 0; i < vector.Length; i++)
            {
                sentenceVector[i] += vector[i];
            }
        }

        for (int i = 0; i < sentenceVector.Length; i++)
        {
            sentenceVector[i] /= words.Length;
        }

        return sentenceVector;
    }
    private double CosineSimilarity(float[] vector1, float[] vector2)
    {
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        for (int i = 0; i < vector1.Length; i++)
        {
            dotProduct += vector1[i] * vector2[i];
            normA += vector1[i] * vector1[i];
            normB += vector2[i] * vector2[i];
        }
        return dotProduct / (Math.Sqrt(normA) * Math.Sqrt(normB));
    }
    #endregion
    #region FastText Cosine Similarity
    private double CalculateFastTextCosineSimilarity(string sentence1, string sentence2)
    {
        var embedding1 = GetFastTextEmbedding(sentence1);
        var embedding2 = GetFastTextEmbedding(sentence2);

        return FastCosineSimilarity(embedding1, embedding2);
    }

    private float[] GetFastTextEmbedding(string sentence)
    {
        var words = sentence.Split(' ');
        var wordVectors = words.Select(word => _fastTextModel.GetWordVector(word)).ToList();

        // Average the word vectors to get a sentence-level embedding
        var sentenceEmbedding = new float[wordVectors[0].Length];
        foreach (var wordVector in wordVectors)
        {
            for (int i = 0; i < wordVector.Length; i++)
            {
                sentenceEmbedding[i] += wordVector[i];
            }
        }

        for (int i = 0; i < sentenceEmbedding.Length; i++)
        {
            sentenceEmbedding[i] /= wordVectors.Count;
        }

        return sentenceEmbedding;
    }

    private double FastCosineSimilarity(float[] vec1, float[] vec2)
    {
        double dotProduct = 0;
        double magnitude1 = 0;
        double magnitude2 = 0;

        for (int i = 0; i < vec1.Length; i++)
        {
            dotProduct += vec1[i] * vec2[i];
            magnitude1 += vec1[i] * vec1[i];
            magnitude2 += vec2[i] * vec2[i];
        }

        return dotProduct / (Math.Sqrt(magnitude1) * Math.Sqrt(magnitude2));
    }
    #endregion
    #region TF-IDF Cosine Similarity
    private double CalculateTfidfCosineSimilarity(string sentence1, string sentence2)
    {
        var sentences = new List<string> { sentence1, sentence2 };
        var tfidfVectors = CalculateTfidfVectors(sentences);

        return CosineSimilarity(tfidfVectors[0], tfidfVectors[1]);
    }

    private List<Dictionary<string, double>> CalculateTfidfVectors(List<string> sentences)
    {
        var termFrequencies = new List<Dictionary<string, double>>();
        var documentFrequency = new Dictionary<string, int>();

        // Calculate Term Frequency (TF) for each sentence
        foreach (var sentence in sentences)
        {
            var termFrequency = new Dictionary<string, double>();
            var words = Tokenize(sentence);

            foreach (var word in words)
            {
                if (!termFrequency.ContainsKey(word))
                {
                    termFrequency[word] = 0;
                }
                termFrequency[word]++;
            }

            foreach (var word in termFrequency.Keys.ToList())
            {
                termFrequency[word] /= words.Length;

                if (!documentFrequency.ContainsKey(word))
                {
                    documentFrequency[word] = 0;
                }
                documentFrequency[word]++;
            }

            termFrequencies.Add(termFrequency);
        }

        // Calculate TF-IDF
        var tfidfVectors = new List<Dictionary<string, double>>();
        int totalDocuments = sentences.Count;

        foreach (var tf in termFrequencies)
        {
            var tfidf = new Dictionary<string, double>();
            foreach (var word in tf.Keys)
            {
                double idf = Math.Log((double)totalDocuments / documentFrequency[word]);
                tfidf[word] = tf[word] * idf;
            }
            tfidfVectors.Add(tfidf);
        }

        return tfidfVectors;
    }

    private string[] Tokenize(string sentence)
    {
        sentence = sentence.ToLower();
        return Regex.Split(sentence, @"\W+");
    }

    private double CosineSimilarity(Dictionary<string, double> vec1, Dictionary<string, double> vec2)
    {
        double dotProduct = 0.0;
        double magnitude1 = 0.0;
        double magnitude2 = 0.0;

        // Calculate dot product and magnitudes
        foreach (var term in vec1.Keys)
        {
            if (vec2.ContainsKey(term))
            {
                dotProduct += vec1[term] * vec2[term];
            }
            magnitude1 += vec1[term] * vec1[term];
        }

        foreach (var term in vec2.Values)
        {
            magnitude2 += term * term;
        }

        // Return cosine similarity
        if (magnitude1 == 0 || magnitude2 == 0)
        {
            return 0.0;
        }
        return dotProduct / (Math.Sqrt(magnitude1) * Math.Sqrt(magnitude2));
    }
    #endregion
    #region BERT Cosine Similarity
    private async Task<double> CalculateBertCosineSimilarity(string sentence1, string sentence2)
    {
        var sentences = new List<string> { sentence1, sentence2 };
        var embeddings = await GetBertEmbeddings(sentences);

        return BertCosineSimilarity(embeddings[0], embeddings[1]);
    }

    private async Task<List<Dictionary<string, double>>> GetBertEmbeddings(List<string> sentences)
    {
        var jsonContent = JsonSerializer.Serialize(new { sentences });
        var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

        var response = await _httpClient.PostAsync("https://your-bert-model-api-endpoint", content);

        if (response.IsSuccessStatusCode)
        {
            var result = await response.Content.ReadAsStringAsync();
            var embeddings = JsonSerializer.Deserialize<List<Dictionary<string, double>>>(result);
            return embeddings;
        }

        throw new Exception("Failed to get BERT embeddings");
    }

    private double BertCosineSimilarity(Dictionary<string, double> vec1, Dictionary<string, double> vec2)
    {
        double dotProduct = 0.0;
        double magnitude1 = 0.0;
        double magnitude2 = 0.0;

        foreach (var term in vec1.Keys)
        {
            if (vec2.ContainsKey(term))
            {
                dotProduct += vec1[term] * vec2[term];
            }
            magnitude1 += vec1[term] * vec1[term];
        }

        foreach (var term in vec2.Values)
        {
            magnitude2 += term * term;
        }

        if (magnitude1 == 0 || magnitude2 == 0)
        {
            return 0.0;
        }
        return dotProduct / (Math.Sqrt(magnitude1) * Math.Sqrt(magnitude2));
    }
    #endregion
    #region Python SentenceTransformer('all-MiniLM-L6-v2') Similarity
    // method for calling Python SentenceTransformer API
    public async Task<double> CalculateSentenceTransformerSimilarity(string sentence1, string sentence2)
    {
        // Prepare request data
        var jsonContent = JsonSerializer.Serialize(new { sentence1, sentence2 });
        var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

        // Call the Python API to get the similarity scores
        var response = await _httpClient.PostAsync("http://localhost:5000/similarity", content);

        if (response.IsSuccessStatusCode)
        {
            var result = await response.Content.ReadAsStringAsync();
            var similarityResult = JsonSerializer.Deserialize<Dictionary<string, double>>(result);
            return similarityResult["similarity"];
        }
        else
        {
            return 0.0;
            throw new Exception("Failed to get similarity score from Python service");
        }
    }
    #endregion
}

