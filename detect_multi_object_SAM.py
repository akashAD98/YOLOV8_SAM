# install required libraries

!pip install -U torch  ultralytics

# import & downloading the sam weights
import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())
import sys
!{sys.executable} -m pip install opencv-python matplotlib
!{sys.executable} -m pip install 'git+https://github.com/facebookresearch/segment-anything.git'
    
!mkdir images
#!wget -P images https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg
#!wget -P images https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/groceries.jpg
        
!wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  
  


from ultralytics import YOLO
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor

def yolov8_detection(model, image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image, stream=True)  # generator of Results objects

    boxes_list = []
    classes_list = []
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        class_id = result.boxes.cls.long().tolist()
        boxes_list.append(boxes.xyxy.tolist())
        classes_list.append(class_id)

    bbox = [[int(i) for i in box] for boxes in boxes_list for box in boxes]
    class_id = [class_id for classes in classes_list for class_id in classes]

    return bbox, class_id, image


model = YOLO("/content/best.pt")
yolov8_boxex,yolov8_class_id, image = yolov8_detection(model, "/content/two-cigarette.jpg")
input_boxes = torch.tensor(yolov8_boxex, device=model.device)

sam_checkpoint = "/content/sam_vit_h_4b8939.pth"
model_type = "vit_h"
#device = "cuda:0"
device ='cpu'
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
predictor.set_image(image)

transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
masks, _, _ = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False,
)

for i, mask in enumerate(masks):

    binary_mask = masks[i].squeeze().cpu().numpy().astype(np.uint8)

    # Find the contours of the mask
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)

    # Get the new bounding box
    bbox = [int(x) for x in cv2.boundingRect(largest_contour)]

    # Get the segmentation mask for object
    segmentation = largest_contour.flatten().tolist()

    # Write bounding boxes to file in YOLO format
# Write bounding boxes to file in YOLO format
    with open("BBOX_Two_cigretee.txt", "a") as f:
        # Get the bounding box coordinates of the largest contour
        x, y, w, h = bbox
        # Convert the coordinates to YOLO format and write to file
        f.write(
            "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(yolov8_class_id[i],
                (x + w / 2) / image.shape[1],
                (y + h / 2) / image.shape[0],
                w / image.shape[1],
                h / image.shape[0],
            )
        )
        f.write("\n")

    mask = segmentation

    # load the image
    # width, height = image_path.size
    height, width = image.shape[:2]

    # convert mask to numpy array of shape (N,2)
    mask = np.array(mask).reshape(-1, 2)

    # normalize the pixel coordinates
    mask_norm = mask / np.array([width, height])

    # compute the bounding box
    xmin, ymin = mask_norm.min(axis=0)
    xmax, ymax = mask_norm.max(axis=0)
    bbox_norm = np.array([xmin, ymin, xmax, ymax])

    # concatenate bbox and mask to obtain YOLO format
    # yolo = np.concatenate([bbox_norm, mask_norm.reshape(-1)])
    yolo = mask_norm.reshape(-1)

    #compute the bounding box
    #write the yolo values to a text file
    with open("yolomask_two_cigretee.txt", "a") as f:
        for val in yolo:
            f.write("{} {:.6f}".format(yolov8_class_id[i],val))
        f.write("\n")

    print("Bounding box:", bbox)
    print("yolo", yolo)
