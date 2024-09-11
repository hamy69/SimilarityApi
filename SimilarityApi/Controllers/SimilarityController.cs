using Microsoft.AspNetCore.Mvc;
using SimilarityApi.Models;
using SimilarityApi.Services;

namespace SimilarityApi.Controllers;
[ApiController]
[Route("[controller]")]
public class SimilarityController : ControllerBase
{
    private readonly ILogger<SimilarityController> _logger;
    private readonly IServiceScopeFactory _scopeFactory;

    public SimilarityController(ILogger<SimilarityController> logger, IServiceScopeFactory scopeFactory)
    {
        _logger = logger;
        _scopeFactory = scopeFactory;
    }

    [HttpPost]
    public async Task<ActionResult<List<SimilarityResponse>>> Post([FromBody] SimilarityRequest request)
    {
        using (var scope = _scopeFactory.CreateScope())
        {
            var similarityService = scope.ServiceProvider.GetRequiredService<ISimilarityService>();

            var similarityTasks = request.Sentences
                .Select(async s => new SimilarityResponse
                {
                    Id = s.Id,
                    Sentence = s.Sentence,
                    Similarities = await similarityService.CalculateSimilarity(s.Sentence, request.SingleSentence)
                }).ToList();

            var bestMatch = (await Task.WhenAll(similarityTasks))
                .OrderByDescending(r => r.Similarities.SentenceTransformer)
                .ToList();

            if (bestMatch != null //&& bestMatch.Any(u => u.Similarities.FastText >= 0.95 || u.Similarities.LevenshteinDistance >= 0.95 || u.Similarities.Jaccard >= 0.95)
                )
            {
                return Ok(bestMatch
                    //.Where(u => u.Similarities.FastText >= 0.95 || u.Similarities.LevenshteinDistance >= 0.95 || u.Similarities.Jaccard >= 0.95)
                    .ToList());
            }

            return NotFound();
        }
    }
}

