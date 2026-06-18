# You must specify the video path in the "video_path" argument below.
## You must also specify the path for the trained model you wish to deploy by adjusting the "model_path" argument below.

from ultralytics import YOLO
import cv2

def draw_yolo_bboxes(image_path, model_path, output_path, bbox_color=(0, 255, 0), label_color=(255, 255, 255), confidence=0.35):
    """
    Runs YOLO object detection and draws bounding boxes with custom colors.

    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the trained YOLO model.
        output_path (str): Path to save the output image.
        bbox_color (tuple): RGB color for the bounding box (default: Green).
        label_color (tuple): RGB color for the label text (default: White).
        confidence (float): Confidence threshold for YOLO detections.
    """
    # Load YOLO model
    model = YOLO(model_path)

    # Run inference
    results = model(image_path, conf=confidence)

    # Process results
    for result in results:
        img = result.orig_img  # Original image
        boxes = result.boxes.xyxy.cpu().numpy()  # Extract bounding box coordinates
        confidences = result.boxes.conf.cpu().numpy()  # Extract confidence scores
        class_ids = result.boxes.cls.cpu().numpy()  # Extract class IDs

        # Get class names if available
        class_names = model.names if hasattr(model, "names") else {}

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
            class_id = int(class_ids[i])
            confidence_score = confidences[i]

            # Label with class name and confidence
            label = f"{class_names.get(class_id, 'Object')} {confidence_score:.2f}"

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, 2)

            # Get text size for label background
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x, text_y = x1, y1 - 5
            text_w, text_h = text_size

            # Draw background for text
            cv2.rectangle(img, (text_x, text_y - text_h - 5), (text_x + text_w, text_y), bbox_color, -1)

            # Draw label text
            cv2.putText(img, label, (text_x, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1, cv2.LINE_AA)

    # Save the output image
    cv2.imwrite(output_path, img)
    print(f"Exported image with bounding boxes to '{output_path}'")

def process_video_yolo(video_path, model_path, output_path, confidence=0.35):
    """
    Runs YOLO object detection on a video and overlays bounding boxes and labels with pre-defined class-specific colors.

    Args:
        video_path (str): Path to the input video.
        model_path (str): Path to the trained YOLO model.
        output_path (str): Path to save the output video.
        confidence (float): Confidence threshold for YOLO detections.
    """
    # Load YOLO model
    model = YOLO(model_path)

    # Define class-specific colors (BGR format)
    class_colors = {
        0: (0, 0, 255),    # red for class 0 (FEED)
        1: (154, 250, 0),    # mediumSpringGreen for class 1 (loco)
        2: (255, 0, 255),    # PURPLE/FUCHSIA for class 2 (obman)
        3: (0, 215, 255),  # gold for class 3 (rest)
        4: (255, 152, 32)   #dodger blue for class 4 (stereo)
    }
    default_bbox_color = (255, 255, 255)  # Default white color for any other class
    default_label_color = (0, 0, 0)  # Black text for visibility

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change to 'XVID' for AVI format
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Run YOLO inference
        results = model(frame, conf=confidence)

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Extract bounding box coordinates
            confidences = result.boxes.conf.cpu().numpy()  # Extract confidence scores
            class_ids = result.boxes.cls.cpu().numpy()  # Extract class IDs

            # Get class names if available
            class_names = model.names if hasattr(model, "names") else {}

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
                class_id = int(class_ids[i])
                confidence_score = confidences[i]

                # Get predefined color for the class, or default if not in dictionary
                bbox_color = class_colors.get(class_id, default_bbox_color)

                # Label with class name and confidence
                label = f"{class_names.get(class_id, 'Object')} {confidence_score:.2f}"

                # Draw thicker bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 6)  # Increased thickness to 6

                # Get text size for label background (double the original size)
                font_scale = 1.0  # Increased font scale
                text_thickness = 2  # Increased thickness
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
                text_x, text_y = x1, y1 - 10
                text_w, text_h = text_size

                # Draw larger background for text
                cv2.rectangle(frame, (text_x, text_y - text_h - 10), (text_x + text_w + 10, text_y), bbox_color, -1)

                # Draw larger label text
                cv2.putText(frame, label, (text_x + 5, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, default_label_color, text_thickness, cv2.LINE_AA)

        # Write frame to output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Processed video saved to '{output_path}'")

# Example usage
process_video_yolo(
    video_path=r"C:\Users\lchen\Desktop\PantherAI - Final Scripts\Vids4Testing\12_5_2024 10_46_02 AM (UTC-05_00).mkv",  # Replace with your video path
    model_path=r"C:\Users\lchen\Desktop\PantherAI - Final Scripts\best_1280.pt",  # Replace with your trained YOLO model path
    output_path="output.mp4",  # Output video path
    confidence=0.5
)


'''
# Example usage
draw_yolo_bboxes(
    image_path="1.jpg",  # Replace with your image path
    model_path="best.pt",  # Replace with your trained YOLO model path
    output_path="output.jpg",  # Output image path
    bbox_color=(255, 0, 0),  # Bounding box color (Blue) because (B,G,R)
    label_color=(0, 255, 255),  # Label text color (Yellow) because (B,G,R)
    confidence=0.5
)
'''