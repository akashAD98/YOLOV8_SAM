from ultralytics import YOLO
import numpy as np
import cv2
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt

def yolov8_detection():

    model=YOLO('/content/best_adam_lr_batch16_az.pt')
    image = "/content/Cigaretee_i_Weapon_Pragati138.jpg"
    
    results = model(image, stream=True)  # generator of Results objects

    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
    
    bbox=boxes.xyxy.tolist()[0]
    return bbox

image = cv2.imread('/content/Cigaretee_i_Weapon_Pragati138.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
predictor.set_image(image)

yolov8_boxex = yolov8_detection()
input_box = np.array(yolov8_boxex)

masks, _, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,
)

for i, mask in enumerate(masks):
    # Convert the mask to a binary image
    #binary_mask = mask.cpu().numpy().squeeze().astype(np.uint8)
    # Convert the mask to a binary image    
    binary_mask = torch.from_numpy(masks).squeeze().numpy().astype(np.uint8)


    # Find the contours of the mask
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour based on area
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the new bounding box
    bbox = [int(x) for x in cv2.boundingRect(largest_contour)]

    # Get the segmentation mask for object 
    segmentation = largest_contour.flatten().tolist()

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    show_box(input_box, plt.gca())
    plt.axis('off')
    plt.show()

    # Print the bounding box and segmentation mask
    print("Bounding box:", bbox)
    print("Segmentation mask:", segmentation)
