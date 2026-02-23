using System;
using System.IO;
using System.Linq;
using OpenCvSharp;

namespace PerchObserver.FeatureExtraction;

public class FeatureBatchRunner : IDisposable
{
    private readonly FeatureExtractor _extractor;
    private readonly FeatureProcessor _processor;
    private bool disposedValue;
    private const int MODEL_INPUT_WIDTH = 224;
    private const int MODEL_INPUT_HEIGHT = 224;

    public FeatureBatchRunner()
    {
        _extractor = new FeatureExtractor(Path.Combine("Data", "dinov2_uint8.onnx"), MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);
        _processor = new FeatureProcessor();
    }

    public void RunBatch(string folderPath)
    {
        foreach (var file in Directory.EnumerateFiles(folderPath, "*png", new EnumerationOptions()
        {
            RecurseSubdirectories = true
        }))
        {
            using var mat = Cv2.ImRead(file);
            using var resized = mat.Resize(new Size(MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT));
            var embeddings = _extractor.ExtractFeatures(resized);
            var birdRecord = _processor.ProcessFeatures(Path.GetFileName(file), embeddings);

            Console.WriteLine($"File: {file}, Assigned Bird ID: {birdRecord.Id}, Similarity Score: {birdRecord.Features.Last().SimilarityScore}");
        }
    }

    #region DISPOSE

    protected virtual void Dispose(bool disposing)
    {
        if (!disposedValue)
        {
            if (disposing)
            {
                // TODO: dispose managed state (managed objects)
                _extractor.Dispose();
            }

            // TODO: free unmanaged resources (unmanaged objects) and override finalizer
            // TODO: set large fields to null
            disposedValue = true;
        }
    }

    // // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
    // ~FeatureBatchRunner()
    // {
    //     // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
    //     Dispose(disposing: false);
    // }

    public void Dispose()
    {
        // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }

    #endregion

}