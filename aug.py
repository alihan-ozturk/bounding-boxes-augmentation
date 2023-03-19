import os
import numpy as np
import cv2 as cv
import albumentations as A
from copy import copy

transform = A.Compose([
    A.RandomCrop(width=300, height=300),
    A.RandomBrightnessContrast(p=1),
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3))

imgPath = "C:/Users/Alihan/Desktop/imgs/"
annoPath = "C:/Users/Alihan/Desktop/labels/"
key = ord("q")


def convert(bboxParams, shape):
    for bboxParam in bboxParams:
        x1 = bboxParam[0] * shape[1] - bboxParam[2] * shape[1] / 2
        y1 = bboxParam[1] * shape[0] - bboxParam[3] * shape[0] / 2
        x2 = x1 + bboxParam[2] * shape[1]
        y2 = y1 + bboxParam[3] * shape[0]
        yield int(x1), int(y1), int(x2), int(y2)


for imgName in os.listdir(imgPath):
    labelPath = annoPath + imgName.split(".")[0] + ".txt"
    img = cv.imread(imgPath + imgName)
    org = copy(img)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    anno = []
    with open(labelPath, "r") as labels:
        for label in labels.readlines():
            anno.append((label[:-1] + " 0").split(" ")[1:])
    anno = np.array(anno, dtype=float)

    transformed = transform(image=img, bboxes=anno)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']

    for x1, y1, x2, y2 in convert(anno, img.shape):
        cv.rectangle(org, (x1, y1), (x2, y2), (255, 0, 0), 1)
    cv.imshow(imgName, org)

    if len(transformed_bboxes) > 0:
        for x1, y1, x2, y2 in convert(transformed_bboxes, transformed_image.shape):
            cv.rectangle(transformed_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            transformed_image = cv.cvtColor(transformed_image, cv.COLOR_RGB2BGR)
            cv.imshow("transformed", transformed_image)
    else:
        print(imgName, "no drones in sight")

    if key == cv.waitKey(0):
        cv.destroyAllWindows()
        break
