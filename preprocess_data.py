import argparse
import os
import numpy as np


class BoundingBox:
    def __init__(self, x_min, y_min, w, h, category):
        self.x_min = x_min
        self.y_min = y_min
        self.w = w
        self.h = h
        self.category = category


class AnnotatedImage:
    def __init__(self, image_path):
        self.image_path = image_path
        # list of class bounding boxes
        self.BoundingBoxes = []


def pre_process_data(tl_data_path):
    # list of all annotated_images in data set
    annotated_images = []
    annotations_dir = os.path.join(tl_data_path, 'Annotations')
    images_dir = os.path.join(tl_data_path, 'PNGImages')
    for filename in os.listdir(annotations_dir):
        image_number = os.path.splitext(os.path.basename(filename))[0]
        image_path = os.path.join(tl_data_path, 'PNGImages', image_number + '.png')
        image = AnnotatedImage(image_path)
        annotation_path = os.path.join(annotations_dir, filename)
        annotation = np.genfromtxt(annotation_path, delimiter='\t', dtype=int)
        # annotation2 = np.genfromtxt(annotation_path, delimiter='\t', dtype=str)
        for index in range(len(annotation)):
            box_num = annotation[index]
            category = box_num
            x_min = box_num
            y_min = box_num
            w = box_num
            h = box_num
            box = BoundingBox(x_min, y_min, w, h, category)
            image.BoundingBoxes.append(box)
        annotated_images.append(image)
    return annotated_images

if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('-d', '--data_path', help='path to TL data')
    #args = parser.parse_args()
    #pre_process_data(args.data_path)

    data_path = '/home/daeyoung-snu/YoloTensorFlow229-modified/LISA_TL_dataset/'
    results = pre_process_data(data_path)
    print results[1].BoundingBoxes[2].category
