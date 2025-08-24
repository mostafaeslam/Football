# Football Computer Vision Task

This project implements a computer vision pipeline for football (soccer) video analysis. The system can detect players, referees, and the ball, differentiate between teams based on jersey colors, track objects across frames, and recognize jersey numbers.

## Features

- **Object Detection**: Detects players, referees, and the ball using YOLOv8
- **Team Differentiation**: Distinguishes between teams using K-means clustering on jersey colors
- **Jersey Number Recognition**: Identifies player numbers using OCR
- **Object Tracking**: Tracks objects across video frames using IOU-based tracking
- **Pitch Detection**: Identifies the playing field and inner pitch area

## Project Structure

```
Football-cv-task/
├── Football_Task.ipynb    # Main notebook with all processing code
├── README.md              # Project documentation
├── yolov8n.pt            # Pre-trained YOLOv8 nano model
├── data/                  # Data directory
│   └── raw/               # Raw input videos
│       └── CV_Task.mkv    # Sample input video
├── models/                # Models directory
│   └── yolov8n.pt        # YOLOv8 model for detection
└── outputs/               # Output directory
    ├── frames/            # Extracted video frames
    ├── logs/              # Processing logs
    └── videos/            # Processed output videos
```

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Ultralytics YOLOv8
- EasyOCR
- PyTorch
- Scikit-learn
- Matplotlib
- PyYAML

## Notebook Contents

The main notebook `Football_Task.ipynb` contains the complete pipeline organized in cells:

1. **Configuration**: Sets up parameters for detection, tracking, and visualization
2. **Environment Setup**: Imports libraries and initializes the environment
3. **Utilities**: Helper functions for processing frames and detecting objects
4. **Object Detection**: YOLOv8 implementation for detecting players, referees, and the ball
5. **Team Differentiation**: Clustering algorithms to distinguish team jerseys
6. **Jersey Number Recognition**: OCR processing to identify player numbers
7. **Object Tracking**: IOU-based tracking across video frames
8. **Preview Loop**: Processes a sample of frames to verify pipeline functionality
9. **Full Video Processing**: Complete pipeline for processing entire videos

## Usage

1. Open the `Football_Task.ipynb` notebook in Jupyter or VS Code
2. Make sure all dependencies are installed
3. Configure the input video path in the CONFIG dictionary
4. Run all cells to process the video
5. Check the `outputs/videos` directory for the annotated video

## Output

The system produces an annotated video with:

- Bounding boxes around detected objects (players, referees, ball)
- Color-coded team identification
- Jersey number annotations when detected
- Object tracking across frames


## Author

Mostafa Eslam
