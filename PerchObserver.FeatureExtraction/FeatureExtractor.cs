using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;

namespace PerchObserver.FeatureExtraction
{
    public class FeatureExtractor : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly int _inputWidth;
        private readonly int _inputHeight;

        protected float[] ChwBuffer { get; set; }

        private readonly DenseTensor<float> _inputTensor;
        private readonly List<NamedOnnxValue> _inputs = new(1);

        private static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
        private static readonly float[] Std = [0.229f, 0.224f, 0.225f];

        private bool _disposed;

        public FeatureExtractor(string modelPath, int inputWidth, int inputHeight)
        {
            _session = new InferenceSession(modelPath);
            _inputWidth = inputWidth;
            _inputHeight = inputHeight;

            ChwBuffer = new float[3 * _inputWidth * _inputHeight];
            _inputTensor = new DenseTensor<float>(ChwBuffer, [1, 3, _inputHeight, _inputWidth]);
        }

        public float[] ExtractFeatures(Mat resizedImg)
        {
            if (resizedImg.Width != _inputWidth || resizedImg.Height != _inputHeight)
                throw new ArgumentException("Image must already be resized to model input size.");

            unsafe
            {
                byte* src = (byte*)resizedImg.DataPointer;
                int hw = _inputWidth * _inputHeight;

                // Convert HWC → CHW + normalize
                for (int i = 0; i < hw; i++)
                {
                    int y = i / _inputWidth;
                    int x = i % _inputWidth;
                    int srcIdx = (y * _inputWidth + x) * 3;

                    float r = src[srcIdx + 2] / 255f;
                    float g = src[srcIdx + 1] / 255f;
                    float b = src[srcIdx + 0] / 255f;

                    ChwBuffer[i] = (r - Mean[0]) / Std[0];
                    ChwBuffer[hw + i] = (g - Mean[1]) / Std[1];
                    ChwBuffer[2 * hw + i] = (b - Mean[2]) / Std[2];
                }
            }

            // Copy CHW buffer → ONNX tensor - FIXED: Access buffer directly instead of creating a copy
            var tensorSpan = _inputTensor.Buffer.Span;
            ChwBuffer.AsSpan().CopyTo(tensorSpan);

            _inputs.Clear();
            _inputs.Add(NamedOnnxValue.CreateFromTensor("pixel_values", _inputTensor));

            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(_inputs);

            var output = results[0].AsTensor<float>();
            return [.. output];
        }


        #region DISPOSE

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // TODO: dispose managed state (managed objects)
                    _session.Dispose();
                }

                // TODO: free unmanaged resources (unmanaged objects) and override finalizer
                // TODO: set large fields to null
                _disposed = true;
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
}
