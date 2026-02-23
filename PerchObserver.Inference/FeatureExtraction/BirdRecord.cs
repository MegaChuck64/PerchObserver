
namespace PerchObserver.Inference.FeatureExtraction;
public class BirdRecord(string id, FeatureSet initialFeature)
{
    public string Id { get; set; } = id;
    public List<FeatureSet> Features { get; set; } = [initialFeature];
}

public class FeatureSet(string id, float[] featureData, float similarityScore)
{
    public string Id { get; set; } = id;
    public float[] Data { get; set; } = featureData;

    public float SimilarityScore { get; set; } = similarityScore;
}
