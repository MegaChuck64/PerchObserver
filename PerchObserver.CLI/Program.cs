using PerchObserver.Inference.FeatureExtraction;
using PerchObserver.Inference.ObjectDetection;
using System.Net.Http.Headers;
using System.Reflection;

namespace PerchObserver.CLI;

internal static class Program
{
    private static string Version = "0.0.0.0";

    private readonly static Dictionary<string, string> ArgDefinitions = new()
    {
        {"-h", "Display help message that lists all commands"},
        {"-help", "Display help message that lists all commands"},
        {"-v",  "Display the current version"},
        {"-run", "Run feature extraction against a folder path. Ex: -run C:\\Pictures\\DataImages" },
    };
    public static void Main(string[] args)
    {
#if DEBUG
        args =
        [
            "-run",
            "Y:\\Videos\\birddata\\FeederCam\\Cardinal"
        ];
#endif

        Version = Assembly.GetExecutingAssembly().GetName().Version?.ToString() ?? "0.0.0.0";

        if (args.Length == 0 || args.Contains("-h") || args.Contains("-help"))
        {
            PrintHelp();
            return;
        }

        if (args.Contains("-v"))
        {
            Console.WriteLine("PerchObserver CLI Version: " + Version);
            return;
        }

        if (args.Contains("-run"))
        {
            Console.WriteLine("PerchObserver CLI Version: " + Version);

            int runIndex = Array.IndexOf(args, "-run");
            if (runIndex + 1 >= args.Length)
            {
                Console.WriteLine("Error: Folder path not specified for -run command.");
                return;
            }
            string folderPath = args[runIndex + 1];
            if (!Directory.Exists(folderPath))
            {
                Console.WriteLine($"Error: The specified folder path does not exist: {folderPath}");
                return;
            }

            using var detector =
                new ObjectDetector(Path.Combine("Data", "yolo11n_320.onnx"), 320, 320, 0.1f, 640, 360);

            var detectionRunner = new ObjectDetectionRunner(detector);

            var videoFiles = Directory.EnumerateFiles(folderPath, "*.mp4", new EnumerationOptions { RecurseSubdirectories = true }).ToList();
            Console.WriteLine($"Found {videoFiles.Count} video files to process");

            foreach (var video in videoFiles)
            {
                Console.WriteLine($"Processing video: {Path.GetFileName(video)}");
                detectionRunner.RunVideo(video, 14); // Class 14 = bird
            }

            return;
        }

    }


    private static void PrintHelp()
    {
        Console.WriteLine("PerchObserver CLI");
        Console.WriteLine("-v " + Version);
        Console.WriteLine("Commands:");
        foreach (var arg in ArgDefinitions)
        {
            Console.WriteLine($"  {arg.Key}: {arg.Value}");
        }
    }
}