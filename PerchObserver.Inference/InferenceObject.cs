using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace PerchObserver.Inference;
public abstract class InferenceObject<T> : IDisposable
{
    private readonly float[] _chwBuffer;
    private readonly string _inputName;
    private readonly InferenceSession _session;
    public int InputWidth { get; private set; }
    public int InputHeight { get; private set; }

    private readonly DenseTensor<float> _inputTensor;
    private readonly List<NamedOnnxValue> _inputs = new(1);

    private bool _disposed;

    public InferenceObject(string modelPath, int inputWidth, int inputHeight, string inputName)
    {
        _session = new InferenceSession(modelPath);
        InputWidth = inputWidth;
        InputHeight = inputHeight;
        _inputName = inputName;
        _chwBuffer = new float[3 * InputWidth * InputHeight];
        _inputTensor = new DenseTensor<float>(_chwBuffer, [1, 3, InputHeight, InputWidth]);
    }

    public T GetResults(Mat resizedImg, string tag)
    {
        if (resizedImg.Width != InputWidth || resizedImg.Height != InputHeight)
            throw new ArgumentException("Image must already be resized to model input size.");

        var inputData = PrepareInputs(resizedImg);
        Array.Copy(inputData, _chwBuffer, inputData.Length);
        _inputs.Clear();
        _inputs.Add(NamedOnnxValue.CreateFromTensor(_inputName, _inputTensor));
        using var results = _session.Run(_inputs);
        var output = results[0].AsTensor<float>();
        return ProcessOutputs(output, tag);
    }

    protected abstract float[] PrepareInputs(Mat resizedImg);
    protected abstract T ProcessOutputs(Tensor<float> output, string tag);

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
    // ~ObjectDetector()
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
}