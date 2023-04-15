import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

from PIL import Image


def yolov8_detection(image):
    model = YOLO('/content/best.pt')
    results = model(image, stream=True)  # generator of Results objects
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
    bbox = boxes.xyxy.tolist()[0]
    return bbox


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


# Load image
# image_path = '/content/Cigaretee_i_Weapon_Pragati138.jpg'
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Load image
image_path = '/content/Cigaretee_i_Weapon_Pragati138.jpg'
image = Image.open(image_path).convert('RGB')


# Detection with YOLOv5
yolov8_boxex = yolov8_detection(image)
input_box = np.array(yolov8_boxex)

# Segment with SAM
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
predictor.set_image(image)

masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,
)

# Show the image with the mask and box overlaid
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(masks[0], plt.gca())
show_box(input_box, plt.gca())
plt.axis('off')
plt.show()
