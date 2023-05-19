# YOLOV8_SAM
yolov8 model with SAM meta


Use yolov8 & SAM model to get segmention for custom model


# installation

```
pip install ultrlytics
pip install 'git+https://github.com/facebookresearch/segment-anything.git'

```

## Download weights 
```
 !wget -P images https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg
 !wget -P images https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/groceries.jpg       
 !wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Test on single Image

```
python main.py
```
## test on single objects

```
python detect_multiple_object_SAM.py
```

## Results

<img width="387" alt="image" src="https://user-images.githubusercontent.com/62583018/233255080-209dec85-44e7-4460-ae9c-bf53b28374ec.png">

<img width="312" alt="image" src="https://user-images.githubusercontent.com/62583018/232183468-d1abeb02-43d0-471a-9fce-e0e40eac69a5.png">

<img width="494" alt="image" src="https://user-images.githubusercontent.com/62583018/233537939-13ff5019-1660-4ee2-ac5f-6bd824312ecc.png">

<img width="493" alt="image" src="https://user-images.githubusercontent.com/62583018/233537998-438bdc78-05b5-4153-8245-9889a52696e2.png">



Bounding box: [478, 1280, 182, 76]

Segmentation mask:
[631, 1280, 630, 1281, 629, 1281, 628, 1282, 626, 1282, 625, 1283, 622, 1283, 621, 1284, 619, 1284, 618, 1285, 615, 1285, 614, 1286, 612, 1286, 611, 1287, 609, 1287, 608, 1288, 607, 1288, 606, 1289, 604, 1289, 603, 1290, 602, 1290, 601, 1291, 599, 1291, 598, 1292, 596, 1292, 595, 1293, 593, 1293, 592, 1294]

### Save the result in yolo format for training Mask segmentation model.

yolo format = [0 0.529687 0 0.014815 0 0.529167 0 0.015741 0 0.525521 0 0.015741 0 0.525000 0 0.016667 0 0.519792 0 0.016667 0 0.519271 0 0.017593 0 0.513021 0 0.017593 0 0.512500 0 0.018519 0 0.505208 0 0.018519]


### TODO


```
- Doing annotations on multiple images  - Done
- Add support for saving annotations in yolo format -Done
- Support jsno format for segmentation model trainig

```

### refrence
```
https://github.com/facebookresearch/segment-anything
https://github.com/ultralytics/ultralytics
````
