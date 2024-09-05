from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

model_path = "weights/person_detect_wts.pt"
img_path = "input_data/whole_image.jpg"

model = YOLO(model_path)

results = model(img_path, conf=0.25, iou=0.4)

for result in results:
  bboxes = result.boxes.xyxy.cpu().numpy()
  # Extract confidence score
  confidences = result.boxes.conf.cpu().numpy()
  # Extract class IDs
  class_ids = result.boxes.cls.cpu().numpy()

  # Open the original image
  image = Image.open(img_path)

  # Create a drawing context
  draw = ImageDraw.Draw(image)

  # Draw each bounding box on the image
  for bbox in bboxes:
      x1, y1, x2, y2 = bbox
      draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

  # Display the image with bounding boxes
  plt.figure(figsize=(10, 10))
  plt.imshow(image)
  plt.axis('off')  # Hide axis
  plt.show()