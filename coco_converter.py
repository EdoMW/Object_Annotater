import os, sys
import json
import numpy as np
from pycocotools import mask
from skimage import measure
from sklearn.model_selection import train_test_split

from config import *


categories = [{"id": 1, "name": "grape", "supercategory": "grapes", }]
licenses = [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 0,
             "name": "Attribution-NonCommercial-ShareAlike License"}]
info = {"description": "COCO 2017 Dataset", "url": "http://cocodataset.org", "version": "1.0", "year": 2017,
        "contributor": "COCO Consortium", "date_created": "2021/07/11"}
annotations = []
images = []
#
# path_masks_npz = r'npzs'
# val_path = r'/content/gdrive/MyDrive/grapes data/Mask_RCNN-master/samples/grape/yolact_copy/images/train'
# masks_dataset = os.listdir(path_masks_npz)
# val_path = os.listdir(val_path)
#
# i = 0
# for im in masks_dataset:
#     i += 1
#     if im[:-4] + '.JPG' not in val_path:
#         continue
#     if im.startswith('.'):
#         continue
#     print(f'image number: {i / len(masks_dataset)}', end='\r')
#     sys.stdout.write("\r" + f'image number: {round(((i / len(masks_dataset))* 100),1)}%')
#     sys.stdout.flush()
#     npy_file = os.path.join(path_masks_npz,
#                             im.split('.')[0] + '.npz')
#     npy_file_2 = os.path.join(path_masks_npz, im)
#     mask = np.load(npy_file)
#     mask_original = mask.f.arr_0
#     img_num = im[:-4]
#     image_name = img_num + ".JPG"
#     image = {
#         "license": 0,
#         "file_name": image_name,
#         # "coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg",
#         "height": mask_original.shape[1],
#         "width": mask_original.shape[0],
#         # "date_captured": "2013-11-14 17:02:52",
#         # "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",
#         "id": int(img_num)
#     }
#
#     images.append(image)
#     for mask_num in range(mask_original.shape[2]):
#
#         ground_truth_binary_mask = mask_original[:, :, mask_num]
#
#         fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
#         encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
#         ground_truth_area = mask.area(encoded_ground_truth)
#         ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
#         contours = measure.find_contours(ground_truth_binary_mask, 0.5)
#
#         object_num = int(img_num) * 10 + mask_num
#         annotation = {
#             "segmentation": [],
#             "iscrowd": 0,
#             "image_id": int(img_num),
#             "category_id": 1,
#             "id": int(object_num),
#             "bbox": ground_truth_bounding_box.tolist(),
#             "area": ground_truth_area.tolist(),
#         }
#
#         for contour in contours:
#             contour = np.flip(contour, axis=1)
#             segmentation = contour.T.ravel().tolist()
#             annotation["segmentation"].append(segmentation)
#         annotations.append(annotation)
#
#
# def check_and_remove_annotations(annotations):
#     a = []
#     id_list = []
#     count = 0
#     for i in range(len(annotations)):
#         if annotations[i]["segmentation"] != []:
#             a.append(annotations[i])
#             count += 1
#         else:
#             id_list.append(annotations[i]["image_id"])
#     return a, id_list
#
#
# def remove_images(images, id_list):
#     b = []
#     for i in range(len(images)):
#         if images[i]["id"] in id_list:
#             continue
#         else:
#             b.append(images[i])
#     return b
#
#
# annotations, images_to_remove = check_and_remove_annotations(annotations)
# print(images_to_remove)
# images = remove_images(images, images_to_remove)
#
# coco = {
#     "info": info,
#     "licenses": licenses,
#     "images": images,
#     "annotations": annotations,
#     "categories": categories
# }
#
# with open('data_coco_train.json', 'w') as fp:
#     json.dump(coco, fp)


def check_convert_annotations():
    if len(os.listdir(r'npzs')) != len(os.listdir(r'images')):
        return True
    return False


if __name__ == "__main__":
    end_program = check_convert_annotations()
    if end_program:
        print("Not all images have been annotated/checked")
    else:
        if not config.pre_allocated:
            X = os.listdir('images')
            y = os.listdir('npzs')
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
            print(X_train, '\n', X_test, '\n', y_train, '\n', y_test)
