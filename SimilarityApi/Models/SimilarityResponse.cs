namespace SimilarityApi.Models;

public class SimilarityResponse
{
    public int Id { get; set; }
    public string Sentence { get; set; }
    public Similarity Similarities { get; set; }
}