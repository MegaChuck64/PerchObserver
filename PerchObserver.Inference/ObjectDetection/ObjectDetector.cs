using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace PerchObserver.Inference.ObjectDetection;

public class ObjectDetector(string modelPath, int inputWidth, int inputHeight, float threshold, int originalWidth, int originalHeight) : 
    InferenceObject<List<Detection>>(modelPath, inputWidth, inputHeight, "images")
{
    private readonly float _threshold = threshold;
    private readonly int _originalWidth = originalWidth;
    private readonly int _originalHeight = originalHeight;
    protected override float[] PrepareInputs(Mat resizedImg)
    {
        unsafe
        {
            var chwBuffer = new float[3 * InputWidth * InputHeight];
            var src = (byte*)resizedImg.DataPointer; // Change from float* to byte*
            int hw = InputHeight * InputWidth;

            Parallel.For(0, hw, i =>
            {
                int y = i / InputWidth;
                int x = i % InputWidth;
                int srcIdx = (y * InputWidth + x) * 3;

                // Convert from byte (0-255) to float (0-1) and reorder BGR to RGB
                chwBuffer[i] = src[srcIdx + 2] / 255.0f;         // R (from B channel)
                chwBuffer[hw + i] = src[srcIdx + 1] / 255.0f;    // G 
                chwBuffer[2 * hw + i] = src[srcIdx] / 255.0f;    // B (from R channel)
            });

            return chwBuffer;
        }
    }


    protected override List<Detection> ProcessOutputs(Tensor<float> output, string tag)
    {
        int rows = output.Dimensions[1]; // 84
        int cols = output.Dimensions[2]; // N (e.g., 8400)
        var dataLen = output.Length;
        
        var outputBuffer = new float[dataLen];

        var arr = output.ToArray();
        if (arr.Length == outputBuffer.Length)
        {
            arr.CopyTo(outputBuffer, 0);
        }
        else
        {
            outputBuffer = arr;
        }

        // Reuse detections list
        var _detections = new List<Detection>();

        for (int i = 0; i < cols; i++)
        {
            float cx = outputBuffer[i];
            float cy = outputBuffer[cols + i];
            float w = outputBuffer[2 * cols + i];
            float h = outputBuffer[3 * cols + i];

            int bestClass = -1;
            float bestScore = 0f;

            for (int c = 4; c < 84; c++)
            {
                float score = outputBuffer[c * cols + i];
                if (score > bestScore)
                {
                    bestScore = score;
                    bestClass = c - 4;
                }
            }

            if (bestScore < _threshold)
                continue;

            float x1 = (cx - w / 2) * _originalWidth / InputWidth;
            float y1 = (cy - h / 2) * _originalHeight / InputHeight;
            float x2 = (cx + w / 2) * _originalWidth / InputWidth;
            float y2 = (cy + h / 2) * _originalHeight / InputHeight;

            _detections.Add(new Detection
            {
                X1 = x1,
                Y1 = y1,
                X2 = x2,
                Y2 = y2,
                Score = bestScore,
                ClassId = bestClass
            });
        }

        _detections = NMS(_detections, 0.45f);

        return _detections;
    }

    //non-maximum suppression | postprocessing to remove overlapping boxes
    private static List<Detection> NMS(List<Detection> dets, float iouThreshold)
    {
        var result = new List<Detection>(dets.Count);
        dets.Sort((a, b) => b.Score.CompareTo(a.Score));
        var suppressed = new bool[dets.Count];

        for (int i = 0; i < dets.Count; i++)
        {
            if (suppressed[i]) continue;
            var best = dets[i];
            result.Add(best);

            for (int j = i + 1; j < dets.Count; j++)
            {
                if (!suppressed[j] && IoU(best, dets[j]) > iouThreshold)
                    suppressed[j] = true;
            }
        }
        return result;
    }

    //intersection over union | measure of overlap between two boxes
    private static float IoU(Detection a, Detection b)
    {
        float x1 = Math.Max(a.X1, b.X1);
        float y1 = Math.Max(a.Y1, b.Y1);
        float x2 = Math.Min(a.X2, b.X2);
        float y2 = Math.Min(a.Y2, b.Y2);

        float inter = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
        float areaA = (a.X2 - a.X1) * (a.Y2 - a.Y1);
        float areaB = (b.X2 - b.X1) * (b.Y2 - b.Y1);

        return inter / (areaA + areaB - inter + 1e-6f);
    }


}