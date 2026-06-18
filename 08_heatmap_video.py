# You must specify the video path in the "video_folder" argument belo (bottom of script).
## You must also specify the path for the trained model you wish to deploy by adjusting the "model_path" argument below.
### If analyzing >1 video, make sure the videos being analyzed are all from the same habitat area; otherwise the heatmap will not be accurately displayed.
#### Individual heatmaps will be produced for each unique behavioural class on which predictions are made. 


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
### parameters below, blue_kernel needs to be an odd integer; set to 1 keeps blur effect smaller ; grid_size (25,125 creates less boxier of shape); influence_radius: when to 10 or larger, mixes/blurs pixel associations

def generate_class_heatmaps_from_videos(
    model_path, video_folder, output_folder, grid_size=(125,200), overlay_alpha=0.6, frame_interval=1, blur_kernel=1, influence_radius = 1
):
    os.makedirs(output_folder, exist_ok=True)
    model = YOLO(model_path)
    heatmaps = {}
    class_names = model.names

    for cls in class_names.values():
        heatmaps[cls] = np.zeros(grid_size)

    video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(('.mkv', '.avi'))]

    if not video_files:
        print("No videos found in the specified folder.")
        return

    total_frames_processed = 0
    first_frame = None

    for video_name in video_files:
        video_path = os.path.join(video_folder, video_name)
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(image)

                for result in results:
                    for box in result.boxes.data.tolist():
                        x_min, y_min, x_max, y_max, conf, cls_id = box
                        cls_name = class_names[int(cls_id)]

                        x_center = int(((x_min + x_max) / 2) / result.orig_shape[1] * grid_size[1])
                        y_center = int(((y_min + y_max) / 2) / result.orig_shape[0] * grid_size[0])

                        if 0 <= x_center < grid_size[1] and 0 <= y_center < grid_size[0]:
                            heatmaps[cls_name][y_center, x_center] += 1

                            # Apply circular influence around detection point
                            for dx in range(-2, 3):  # Spread radius
                                for dy in range(-2, 3):
                                    nx, ny = x_center + dx, y_center + dy
                                    if 0 <= nx < grid_size[1] and 0 <= ny < grid_size[0]:
                                        distance = np.sqrt(dx**2 + dy**2)
                                        heatmaps[cls_name][ny, nx] += max(0, 1 - distance / 3)

                total_frames_processed += 1  # Increment frame counter
                if first_frame is None:
                    first_frame = Image.fromarray(image)
                    first_frame_array = np.array(first_frame.convert("L")) / 255.0
                    first_frame_gray_rgb = np.stack([first_frame_array] * 3, axis=-1)

            frame_idx += 1
        cap.release()

    if first_frame is None:
        print("No valid frames extracted from videos.")
        return

    orig_width, orig_height = first_frame.size

    # Define the ROYGBIV colormap (using rainbow)
    roygbiv = plt.cm.get_cmap("rainbow")

    for cls_name, heatmap in heatmaps.items():
        # Normalize and apply Gaussian blur
        heatmap_normalized = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
        heatmap_blurred = cv2.GaussianBlur(heatmap_normalized, (blur_kernel, blur_kernel), 0)

        # Apply 'rainbow' colormap for intensity
        heatmap_colored = roygbiv(heatmap_blurred)[:, :, :3]
        heatmap_resized = Image.fromarray((heatmap_colored * 255).astype(np.uint8)).resize(
            (orig_width, orig_height), Image.Resampling.BILINEAR
        )
        heatmap_resized_array = np.array(heatmap_resized) / 255.0

        # Set 0 values to transparent (exclude from overlay)
        heatmap_resized_array[heatmap_resized_array == 0] = np.nan

        # Prepare overlay with transparent pixels where no detection exists
        overlay = first_frame_gray_rgb.copy()

        # Apply heatmap overlay where heatmap is not NaN (detection exists)
        for i in range(overlay.shape[0]):
            for j in range(overlay.shape[1]):
                if not np.isnan(heatmap_resized_array[i, j]).any():
                    overlay[i, j] = overlay_alpha * heatmap_resized_array[i, j] + (1 - overlay_alpha) * first_frame_gray_rgb[i, j]

        overlay_image = Image.fromarray((overlay * 255).astype(np.uint8))

        # Create the output image with legend space
        output_image = Image.new("RGB", (orig_width + 300, orig_height), "white")
        output_image.paste(overlay_image, (0, 0))

        # Add legend using matplotlib and ensuring it corresponds to the 'rainbow' colormap
        fig, ax = plt.subplots(figsize=(5, 5))
        cax = ax.imshow(heatmap_resized, cmap=roygbiv)
        cbar = fig.colorbar(cax, ax=ax, orientation = 'horizontal')
        cbar.set_label('Detection Intensity', rotation=0, labelpad=15) 
	
	# set rotation = 270 if wnt it vertic 
        # Save the legend as an image
        legend_path = os.path.join(output_folder, f"legend_{cls_name}.png")
        fig.savefig(legend_path, bbox_inches='tight', transparent=False)
        plt.close(fig)
	
	# change transparent = TRUE if want legend background transparent instead of white
        # Paste the legend next to the heatmap image
        legend_img = Image.open(legend_path)
        output_image.paste(legend_img, (orig_width, 0))

        # Add metadata text
        draw = ImageDraw.Draw(output_image)
        font = ImageFont.load_default()
        metrics_text = f"Class: {cls_name}\nTotal Detections: {int(np.sum(heatmap))}\nFrames Processed: {total_frames_processed}"
        draw.multiline_text((orig_width + 10, 10), metrics_text, fill="black", font=font)

        # Save the final output image
        output_path = os.path.join(output_folder, f"heatmap_{cls_name}.png")
        output_image.save(output_path)
        print(f"Saved heatmap overlay for class '{cls_name}' to: {output_path}")

# Example usage
model_path = r'C:\Users\lchen\Desktop\addTestData\best_balanced_wAug.pt'
video_folder = r'C:\Users\lchen\Desktop\PantherAI - Final Scripts\Vids4Testing'
output_folder = 'heatmap_output'
generate_class_heatmaps_from_videos(model_path, video_folder, output_folder)
