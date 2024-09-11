namespace SimilarityApi.Models;

public class Similarity
{
    public double LevenshteinDistance { get; set; }
    public double Jaccard { get; set; }
    public double FastText { get; set; }
    public double TfidfCosine { get; set; }
    public double SentenceTransformer { get; set; }
}