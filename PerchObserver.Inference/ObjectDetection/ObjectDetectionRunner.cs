using OpenCvSharp;

namespace PerchObserver.Inference.ObjectDetection;

public class ObjectDetectionRunner : InferenceRunner<List<Detection>>
{
    private readonly string[] _names;
    private readonly Window _window;
    int croppedCount = 0;

    public ObjectDetectionRunner(InferenceObject<List<Detection>> inferenceObject) : base(inferenceObject)
    {
        _names = File.ReadAllLines(Path.Combine("ObjectDetection", "coco.names"));
        _window = new Window();
    }

    protected override void Run(Mat resizedImg, string path)
    {
        var detections = InferenceObject.GetResults(resizedImg, path);
        Console.WriteLine($"File: {path}, Detections Count: {detections.Count}");
        foreach (var detection in detections)
        {
            Console.WriteLine($" - Label: {_names[detection.ClassId]}, Confidence: {detection.Score}, Bounds: [{detection.X1}, {detection.Y1}, {detection.X2}, {detection.Y2}]");
        }
    }

    public void RunVideo(string videoFilePath, params int[] classIdFilters)
    {
        using var capture = new VideoCapture(videoFilePath);
        if (!capture.IsOpened())
        {
            Console.WriteLine($"Error: Could not open video file: {videoFilePath}");
            return;
        }

        int frameWidth = capture.FrameWidth;
        int frameHeight = capture.FrameHeight;
        int totalFrames = (int)capture.Get(VideoCaptureProperties.FrameCount);
        Console.WriteLine($"Processing video: {videoFilePath}, Total Frames: {totalFrames}, Resolution: {frameWidth}x{frameHeight}");

        using var frame = new Mat();
        int frameIndex = 0;

        while (true)
        {
            bool hasFrame = capture.Read(frame);
            if (!hasFrame || frame.Empty())
                break;

            // Resize frame but keep it as uint8 (byte) format for the ObjectDetector
            using var resizedFrame = new Mat();
            Cv2.Resize(frame, resizedFrame, new Size(InferenceObject.InputWidth, InferenceObject.InputHeight));

            var detections = InferenceObject.GetResults(resizedFrame, $"{videoFilePath}_frame{frameIndex}");

            Console.WriteLine($"Video: {videoFilePath}, Frame: {frameIndex}, Detections Count: {detections.Count}");

            if (detections.Count == 0 || !detections.Any(y => classIdFilters.Contains(y.ClassId)))
            {
                frameIndex++;

                // Still show the frame even without detections
                _window.ShowImage(frame);

                // Add WaitKey for window updates and ESC key handling
                int ky = Cv2.WaitKey(1);
                if (ky == 27) // ESC key
                    break;

                continue;
            }

            var filtered = detections.Where(t => classIdFilters.Contains(t.ClassId)).ToList();

            foreach (var detection in filtered)
            {
                croppedCount++;
                Console.WriteLine($" - Label: {_names[detection.ClassId]}, Confidence: {detection.Score}, Bounds: [{detection.X1}, {detection.Y1}, {detection.X2}, {detection.Y2}]");
                Console.WriteLine("Cropped: " + croppedCount);
            }

            DrawDetections(filtered, frame);
            _window.ShowImage(frame);

            // Critical: Add WaitKey for window updates and ESC key handling
            int key = Cv2.WaitKey(1);
            if (key == 27) // ESC key to exit
                break;

            frameIndex++;
        }

        _window.Close();
        Console.WriteLine("Video processing completed.");
        Console.WriteLine("Total cropped: " + croppedCount);
    }


    public void DrawDetections(List<Detection> detections, Mat frame)
    {
        foreach (var det in detections)
        {
            var rect = new Rect(
                (int)det.X1,
                (int)det.Y1,
                (int)(det.X2 - det.X1),
                (int)(det.Y2 - det.Y1)
            );

            var rectColor = Scalar.Red;
            if (det.Score > 0.85f)
                rectColor = Scalar.LimeGreen;
            else if (det.Score > 0.6f)
                rectColor = Scalar.Orange;
            else if (det.Score > 0.4f)
                rectColor = Scalar.Yellow;

            Cv2.Rectangle(frame, rect, rectColor, 2);

            string label = $"{_names[det.ClassId]} {det.Score:0.00}";
            DrawText(frame, label, new Point(rect.X, rect.Y - 5), Scalar.Yellow, 0.5f);
        }
    }

    public static void DrawText(Mat frame, string text, Point location, Scalar color, float scale, int thickness = 2) =>
        Cv2.PutText(
            frame,
            text,
            location,
            HersheyFonts.HersheySimplex,
            scale,
            color,
            thickness);
}
