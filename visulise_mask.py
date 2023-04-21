import cv2
import numpy as np

image = cv2.imread("image.jpg")
h, w = image.shape[:2]
with open("/content/yolomask_forma.txt") as f:
    segment = [np.array(x.split(), dtype=np.float32).reshape(-1, 2) for x in f.read().strip().splitlines() if len(x)]
for s in segment:
    s[:, 0] *= w
    s[:, 1] *= h
cv2.drawContours(image, [s.astype(np.int32) for s in segment], -1, (0, 0, 255))
cv2.imwrite("output_maskimg.jpg", image)
