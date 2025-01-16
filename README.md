# Karate Pose Analysis

A web application that analyzes karate kata videos using pose detection. The application processes videos to:
- Detect and track body poses
- Visualize skeletal movement
- Track center of gravity (tanden)
- Analyze stance and posture

## Features
- Upload and process MP4 videos
- Real-time pose detection and tracking
- Two output videos:
  - Processed video with pose overlay and darkened background
  - Skeleton-only visualization on black background
- Progress tracking during video processing
- Center of gravity (tanden) visualization

## Requirements
- Python 3.8+
- OpenCV
- MediaPipe
- Flask
- NumPy

## Installation
1. Clone the repository: