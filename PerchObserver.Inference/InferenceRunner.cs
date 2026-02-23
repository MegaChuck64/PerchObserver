using OpenCvSharp;

namespace PerchObserver.Inference;

public abstract class InferenceRunner<T>(InferenceObject<T> inferenceObject)
{
    public InferenceObject<T> InferenceObject { get; private set; } = inferenceObject;

    public void RunResizedImg(Mat resizedImg, string imgPath)
    {
        Run(resizedImg, imgPath);
    }

    public void RunRawImg(Mat rawImg, string imgPath)
    {
        using var resizedImg = rawImg.Resize(new Size(InferenceObject.InputWidth, InferenceObject.InputHeight));
        Run(resizedImg, imgPath);
    }

    public void RunRawImgPath(string imgPath)
    {
        using var rawImg = Cv2.ImRead(imgPath);
        using var resizedImg = rawImg.Resize(new Size(InferenceObject.InputWidth, InferenceObject.InputHeight));
        Run(resizedImg, imgPath);
    }

    public void RunRawImgsPathBatch(IEnumerable<string> imgPaths)
    {
        foreach (var imgPath in imgPaths)
        {
            using var rawImg = Cv2.ImRead(imgPath);
            using var resizedImg = rawImg.Resize(new Size(InferenceObject.InputWidth, InferenceObject.InputHeight));
            Run(resizedImg, imgPath);
        }
    }
    protected abstract void Run(Mat resizedImg, string path);
}