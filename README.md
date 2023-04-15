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

## Test on Image

```
python main.py
```


## Results
<img width="312" alt="image" src="https://user-images.githubusercontent.com/62583018/232183468-d1abeb02-43d0-471a-9fce-e0e40eac69a5.png">



### TODO

```
-More flexible code
-Add support for saving annotations in yolo format
```

```
refrence : https://github.com/facebookresearch/segment-anything
https://github.com/ultralytics/ultralytics
````
