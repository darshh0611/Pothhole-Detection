import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
from base64 import b64encode
import tempfile

def save_uploaded_file(uploaded_file):
    upload_dir = "uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def draw_detections(image, detections):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 16)  # Replace with your desired font and size
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        conf, cls_id = det[4:]
        label = f'{int(cls_id)} {conf:.2f}'
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        text_width = font.getlength(label)
        text_height = font.getbbox(label)[3]  # Get the height from the bounding box
        draw.rectangle([x1, y1, x1 + text_width, y1 + text_height + 2], fill=(255, 0, 0), outline=(255, 0, 0))
        draw.text((x1, y1), label, font=font, fill=(255, 255, 255))
    return image

def generate_roadmap(image, detections, threshold=5):
    road_blocked = len(detections) > threshold
    if road_blocked:
        draw = ImageDraw.Draw(image)
        draw.line([(0, 0), (image.width, image.height)], fill=(255, 0, 0), width=10)
        draw.text((10, 10), "Road Blocked", font=ImageFont.truetype("arial.ttf", 36), fill=(255, 0, 0))
    return image, road_blocked

st.title('Pothole Detection App')

# Model selection
model_version = st.selectbox('Choose your model:', ('yolov8n', 'yolov8m'))
model_paths = {
    'yolov8n': 'D:\\IACV_mp\\content\\runs\\detect\\train5\\weights\\best.pt',
    'yolov8m': 'D:\\IACV_mp\\content\\runs\\detect\\train6\\weights\\best.pt'
}
model_path = model_paths[model_version]
model = YOLO(model_path)

uploaded_file = st.file_uploader("Upload an image or video...", type=['jpg', 'jpeg', 'png', 'mp4'])

if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)

    if uploaded_file.name.endswith(('.jpg', '.jpeg', '.png')):
        results = model(file_path, conf=0.25)

        # Iterate over the results and display each processed image
        for result in results:
            result_image = result.orig_img
            st.image(result_image, caption='Processed Image', use_column_width=True)

            detections = result.boxes.data.tolist()
            if detections:
                annotated_image = Image.fromarray(result_image)
                annotated_image = draw_detections(annotated_image, detections)
                roadmap_image, road_blocked = generate_roadmap(annotated_image, detections, threshold=5)
                st.image(roadmap_image, caption='Roadmap', use_column_width=True)
                if road_blocked:
                    st.warning("Too many potholes detected. Road is blocked for cars.")
                else:
                    st.success("Road is clear for cars to drive through.")

    elif uploaded_file.name.endswith('.mp4'):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            st.error("Failed to open the video file.")
        else:
            st.write("Processing video...")
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_bar = st.progress(0)

            # Create a temporary directory for storing processed frames
            with tempfile.TemporaryDirectory() as temp_dir:
                processed_frames = []

                for frame_idx in range(frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model(frame, conf=0.25)

                    for result in results:
                        detections = result.boxes.data.tolist()
                        if detections:
                            result_image = result.orig_img
                            annotated_image = Image.fromarray(result_image)
                            annotated_image = draw_detections(annotated_image, detections)
                            roadmap_image, road_blocked = generate_roadmap(annotated_image, detections, threshold=5)
                            st.image(roadmap_image, caption='Processed Frame', use_column_width=True)
                            if road_blocked:
                                st.warning("Too many potholes detected. Road is blocked for cars.")

                            # Save the processed frame to the temporary directory
                            frame_path = os.path.join(temp_dir, f"frame_{frame_idx}.jpg")
                            roadmap_image.save(frame_path)
                            processed_frames.append(frame_path)

                    progress_bar.progress((frame_idx + 1) / frame_count)

                cap.release()

                # Combine the processed frames into a video
                output_video_path = os.path.join(temp_dir, "output_video.mp4")
                frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (int(frame_width), int(frame_height)))

                for frame_path in processed_frames:
                    frame = cv2.imread(frame_path)
                    video_writer.write(frame)

                video_writer.release()

                # Display the combined video
                video_bytes = open(output_video_path, "rb").read()
                st.video(video_bytes)

    # Cleanup: Remove the uploaded file to clear space
    os.remove(file_path)