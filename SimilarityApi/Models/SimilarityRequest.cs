namespace SimilarityApi.Models;

public class SimilarityRequest
{
    public List<SentenceData> Sentences { get; set; }
    public string SingleSentence { get; set; }
}