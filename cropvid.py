import numpy as np
import cv2
input_video_path = "./mic_video.mp4"
output_video_path = "./mic_video_cropped.mp4"




def crop_video_to_circle(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Create a circular mask
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    radius_reduction_factor = 0.53  # Reduce the circle's radius to 70% of the original
    radius = int(min(width, height) // 2 * radius_reduction_factor)
    cv2.circle(mask, (width//2, height//2), radius, (255, 255, 255), -1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply the mask
        frame = cv2.bitwise_and(frame, mask)

        # Write the frame
        out.write(frame)

    cap.release()
    out.release()
    
crop_video_to_circle(input_video_path, output_video_path)