import cv2
import mediapipe as mp
import os
import numpy as np
import logging
from datetime import datetime

class PoseDetector:
    def __init__(self):
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        
        try:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            self.logger.error(f"Error initializing MediaPipe: {str(e)}")
            raise

        self.progress_callback = None

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def process_video(self, video_path):
        self.logger.info(f"Starting video processing: {video_path}")
        
        if not os.path.exists(video_path):
            self.logger.error(f"Video file not found: {video_path}")
            raise FileNotFoundError(f"Video file not found at {video_path}")

        try:
            # Create output directory with timestamp
            video_filename = os.path.basename(video_path)
            base_name = os.path.splitext(video_filename)[0]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir_name = f"{base_name}_{timestamp}"
            
            downloads_path = os.path.expanduser("~/Downloads")
            output_dir = os.path.join(downloads_path, output_dir_name)
            
            # Create the directory
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"Created output directory: {output_dir}")

            # Setup output paths in the new directory
            processed_path = os.path.join(output_dir, f"{base_name}_processed.mp4")
            skeleton_path = os.path.join(output_dir, f"{base_name}_skeletonOnly.mp4")

            self.logger.info(f"Output paths:\nProcessed: {processed_path}\nSkeleton: {skeleton_path}")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error("Failed to open video file")
                raise IOError("Failed to open video file")

            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            self.logger.info(f"Video properties: {frame_width}x{frame_height} @ {fps}fps")

            # Get total frames for progress calculation
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.logger.info(f"Total frames to process: {total_frames}")

            # Create video writers
            try:
                processed_out = cv2.VideoWriter(
                    processed_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (frame_width, frame_height)
                )
                
                skeleton_out = cv2.VideoWriter(
                    skeleton_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (frame_width, frame_height)
                )
                
                if not processed_out.isOpened() or not skeleton_out.isOpened():
                    raise IOError("Failed to create video writers")
                
            except Exception as e:
                self.logger.error(f"Error creating video writers: {str(e)}")
                raise

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                progress = (frame_count / total_frames) * 100
                if self.progress_callback:
                    self.progress_callback(progress)

                try:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.pose.process(image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    # Create black overlay and skeleton background
                    black_overlay = np.zeros(image.shape, dtype=np.uint8)
                    black_bg = np.zeros(image.shape, dtype=np.uint8)
                    
                    # Create darkened image outside the if block
                    darkened_image = cv2.addWeighted(image, 0.9, black_overlay, 0.1, 0)
                    
                    if results.pose_landmarks:
                        # Calculate tanden position (between hip joints)
                        left_hip = results.pose_landmarks.landmark[23]  # Left hip
                        right_hip = results.pose_landmarks.landmark[24] # Right hip
                        tanden_x = int(((left_hip.x + right_hip.x) / 2) * image.shape[1])
                        tanden_y = int(((left_hip.y + right_hip.y) / 2 + 0.02) * image.shape[0])
                        
                        # Draw landmarks on both outputs
                        self.mp_drawing.draw_landmarks(
                            darkened_image,
                            results.pose_landmarks,
                            self.mp_pose.POSE_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                            self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                        )
                        
                        self.mp_drawing.draw_landmarks(
                            black_bg,
                            results.pose_landmarks,
                            self.mp_pose.POSE_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                            self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                        )
                        
                        # Draw tanden marker on both outputs
                        cv2.circle(darkened_image, (tanden_x, tanden_y), 8, (0, 255, 0), -1)
                        cv2.circle(black_bg, (tanden_x, tanden_y), 8, (0, 255, 0), -1)

                    processed_out.write(darkened_image)
                    skeleton_out.write(black_bg)

                except Exception as e:
                    self.logger.error(f"Error processing frame {frame_count}: {str(e)}")
                    raise

            self.logger.info(f"Processed {frame_count} frames")

            # Ensure we show 100% at the end
            if self.progress_callback:
                self.progress_callback(100)

        except Exception as e:
            self.logger.error(f"Error during video processing: {str(e)}")
            raise

        finally:
            # Clean up
            if 'cap' in locals():
                cap.release()
            if 'processed_out' in locals():
                processed_out.release()
            if 'skeleton_out' in locals():
                skeleton_out.release()
            cv2.destroyAllWindows()
            self.pose.close()

        self.logger.info("Video processing completed successfully")

if __name__ == "__main__":
    detector = PoseDetector()
    video_path = "../data/videos/KataFukyugataIchi.mp4"
    detector.process_video(video_path) 