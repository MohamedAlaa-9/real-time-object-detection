import cv2
import os

def ingest_video(video_path):
    """
    Ingests a video and yields individual frames.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        raise Exception(f"Could not open video: {video_path}")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        yield frame

    video_capture.release()

if __name__ == '__main__':
    # Example usage:
    try:
        video_path = 'datasets/raw/example.mp4'  # Replace with your video path
        frame_generator = ingest_video(video_path)
        
        # Display the first few frames (for testing)
        for i, frame in enumerate(frame_generator):
            cv2.imshow(f'Frame {i}', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to quit
                break
            if i > 5:
                break

        cv2.destroyAllWindows()

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(e)
