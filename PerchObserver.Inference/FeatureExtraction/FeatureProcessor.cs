namespace PerchObserver.Inference.FeatureExtraction;

public class FeatureProcessor
{
    private readonly Dictionary<string, BirdRecord> _birds = [];
    private int _nextId = 0;

    public const float SimilarityThreshold = 0.85f; // Back to reasonable threshold
    public const int MaxSamplesPerBird = 50; // Limit samples to prevent memory bloat

    public BirdRecord ProcessFeatures(string photoId, float[] features)
    {
        var normalizedFeatures = NormalizeFeatures(features);

        // Skip processing if we've already seen this exact image (duplicate detection)
        foreach (var bird in _birds.Values)
        {
            foreach (var embed in bird.Features)
            {
                if (embed.Id == photoId)
                {
                    Console.WriteLine($"Warning: Duplicate image detected: {photoId}");
                    return bird;
                }
            }
        }

        string bestMatchId = "unknown";
        float bestSimilarity = 0.0f;

        // Find the best matching bird using a more sophisticated approach
        foreach (var bird in _birds.Values)
        {
            float similarity = CalculateBirdSimilarity(normalizedFeatures, bird);

            if (similarity > bestSimilarity)
            {
                bestSimilarity = similarity;
                bestMatchId = bird.Id;
            }
        }

        var newSet = new FeatureSet(photoId, normalizedFeatures, bestSimilarity);

        // If we found a good match, add to existing bird
        if (bestSimilarity >= SimilarityThreshold && bestMatchId != "unknown")
        {
            var matchedBird = _birds[bestMatchId];

            // Manage samples: if too many, remove oldest ones
            if (matchedBird.Features.Count >= MaxSamplesPerBird)
            {
                // Remove oldest samples, keep the most recent ones
                var samplesToKeep = matchedBird.Features
                    .Skip(matchedBird.Features.Count - MaxSamplesPerBird + 10) // Keep last 40, make room for 10 more
                    .ToList();
                matchedBird.Features.Clear();
                matchedBird.Features.AddRange(samplesToKeep);
            }

            matchedBird.Features.Add(newSet);
            return matchedBird;
        }
        else
        {
            // Create new bird - set similarity to 0 for new birds
            var newId = $"bird_{_nextId++}";
            var newBirdSet = new FeatureSet(photoId, normalizedFeatures, 0.0f);
            var newBird = new BirdRecord(newId, newBirdSet);
            _birds[newId] = newBird;
            return newBird;
        }
    }

    private static float CalculateBirdSimilarity(float[] features, BirdRecord bird)
    {
        if (bird.Features.Count == 0) return 0f;

        // Strategy: Use top 75th percentile of similarities to be robust against outliers
        var similarities = new List<float>();

        foreach (var embed in bird.Features)
        {
            var sim = ComputeCosineSimilarity(features, embed.Data);
            sim = Math.Min(1.0f, Math.Max(-1.0f, sim));
            similarities.Add(sim);
        }

        similarities.Sort((a, b) => b.CompareTo(a)); // Sort descending

        // Take the top 75% of similarities and average them
        int topCount = Math.Max(1, (int)(similarities.Count * 0.75f));
        float topAverage = similarities.Take(topCount).Average();

        // Give slight bonus if we have many samples (more confidence)
        float confidenceBonus = Math.Min(0.05f, bird.Features.Count * 0.001f);

        return Math.Min(1.0f, topAverage + confidenceBonus);
    }

    public static float[] NormalizeFeatures(float[] features)
    {
        float sumSquares = 0f;
        foreach (var val in features)
        {
            sumSquares += val * val;
        }
        float norm = (float)Math.Sqrt(sumSquares);
        if (norm == 0f) return features; // Zero vector case

        // Check if already normalized (norm very close to 1.0)
        if (Math.Abs(norm - 1.0f) < 1e-6f) return features;

        float[] normalized = new float[features.Length];
        for (int i = 0; i < features.Length; i++)
        {
            normalized[i] = features[i] / norm;
        }
        return normalized;
    }

    public static float ComputeCosineSimilarity(float[] features1, float[] features2)
    {
        if (features1.Length != features2.Length)
            throw new ArgumentException("Feature vectors must be of the same length.");

        float dotProduct = 0f;
        for (int i = 0; i < features1.Length; i++)
        {
            dotProduct += features1[i] * features2[i];
        }

        // Ensure result is within valid cosine similarity range
        return Math.Min(1.0f, Math.Max(-1.0f, dotProduct));
    }

    // Optional: Method to get clustering statistics
    public void PrintClusteringStats()
    {
        Console.WriteLine($"\n=== Bird Clustering Statistics ===");
        Console.WriteLine($"Total birds identified: {_birds.Count}");
        foreach (var bird in _birds.Values)
        {
            Console.WriteLine($"Bird {bird.Id}: {bird.Features.Count} images");
        }
        Console.WriteLine("==================================\n");
    }
}




