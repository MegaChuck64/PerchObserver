using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace PerchObserver.Inference.FeatureExtraction;
public class FeatureExtractor(string modelPath, int inputWidth, int inputHeight) : InferenceObject<BirdRecord>(modelPath, inputWidth, inputHeight, "pixel_values")
{
    private static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] Std = [0.229f, 0.224f, 0.225f];

    private readonly FeatureProcessor _processor = new ();
    protected override float[] PrepareInputs(Mat resizedImg)
    {
        if (resizedImg.Width != InputWidth || resizedImg.Height != InputHeight)
            throw new ArgumentException("Image must already be resized to model input size.");
        var chwBuffer = new float[3 * InputWidth * InputHeight];
        unsafe
        {
            byte* src = (byte*)resizedImg.DataPointer;
            int hw = InputWidth * InputHeight;

            // Convert HWC → CHW + normalize
            for (int i = 0; i < hw; i++)
            {
                int y = i / InputWidth;
                int x = i % InputWidth;
                int srcIdx = (y * InputWidth + x) * 3;

                float r = src[srcIdx + 2] / 255f;
                float g = src[srcIdx + 1] / 255f;
                float b = src[srcIdx + 0] / 255f;

                chwBuffer[i] = (r - Mean[0]) / Std[0];
                chwBuffer[hw + i] = (g - Mean[1]) / Std[1];
                chwBuffer[2 * hw + i] = (b - Mean[2]) / Std[2];
            }
        }
        return chwBuffer;
    }

    protected override BirdRecord ProcessOutputs(Tensor<float> output, string tag)
    {
        return _processor.ProcessFeatures(tag, [.. output]);
    }
}

