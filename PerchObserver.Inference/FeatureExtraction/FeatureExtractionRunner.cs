using OpenCvSharp;
namespace PerchObserver.Inference.FeatureExtraction;

public class FeatureExtractionRunner(InferenceObject<BirdRecord> inferenceObject) : InferenceRunner<BirdRecord>(inferenceObject)
{
    protected override void Run(Mat resizedImg, string imgPath)
    {
        var birdRecord = InferenceObject.GetResults(resizedImg, imgPath);

        Console.WriteLine($"File: {imgPath}, Assigned Bird ID: {birdRecord.Id}, Similarity Score: {birdRecord.Features.Last().SimilarityScore}");
    }
}