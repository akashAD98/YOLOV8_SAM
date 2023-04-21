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
## test on multiple objects

```
python detect_multiple_object.py
```

## Results

<img width="387" alt="image" src="https://user-images.githubusercontent.com/62583018/233255080-209dec85-44e7-4460-ae9c-bf53b28374ec.png">

<img width="312" alt="image" src="https://user-images.githubusercontent.com/62583018/232183468-d1abeb02-43d0-471a-9fce-e0e40eac69a5.png">

<img width="494" alt="image" src="https://user-images.githubusercontent.com/62583018/233537939-13ff5019-1660-4ee2-ac5f-6bd824312ecc.png">

<img width="493" alt="image" src="https://user-images.githubusercontent.com/62583018/233537998-438bdc78-05b5-4153-8245-9889a52696e2.png">



Bounding box: [478, 1280, 182, 76]
Segmentation mask: [631, 1280, 630, 1281, 629, 1281, 628, 1282, 626, 1282, 625, 1283, 622, 1283, 621, 1284, 619, 1284, 618, 1285, 615, 1285, 614, 1286, 612, 1286, 611, 1287, 609, 1287, 608, 1288, 607, 1288, 606, 1289, 604, 1289, 603, 1290, 602, 1290, 601, 1291, 599, 1291, 598, 1292, 596, 1292, 595, 1293, 593, 1293, 592, 1294, 590, 1294, 589, 1295, 587, 1295, 586, 1296, 584, 1296, 583, 1297, 579, 1297, 578, 1298, 576, 1298, 575, 1299, 574, 1299, 573, 1300, 571, 1300, 570, 1301, 569, 1301, 568, 1302, 566, 1302, 565, 1303, 563, 1303, 562, 1304, 561, 1304, 560, 1305, 558, 1305, 557, 1306, 555, 1306, 554, 1307, 552, 1307, 551, 1308, 548, 1308]

Save the result in yolo format for training Mask segmentation model.



### TODO


```
- Doing annotations on multiple images  - Done
- Add support for saving annotations in yolo format
- Support jsno format for segmentation model trainig

```

### refrence
```
https://github.com/facebookresearch/segment-anything
https://github.com/ultralytics/ultralytics
````
