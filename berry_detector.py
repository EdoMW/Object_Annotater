from matplotlib import image
from matplotlib import pyplot
"""
Displaying images with masks on top of them of the experiment
"""

import random

import cv2 as cv
import sys
import os
import warnings
import matplotlib.pyplot as plt
from termcolor import colored
import glob, shutil
import numpy as np
from random import randint

# Next 2 lines are for the test set
# masks_path = r'D:\Users\NanoProject\old_experiments\exp_data_13_46\masks'
# img_path = r'D:\Users\NanoProject\old_experiments\exp_data_13_46\rgb_images\resized'


def show_in_moved_window(win_name, img, i=None, x=0, y=0):  # lab
    """
    show image
    :param win_name: name of the window
    :param img: image to display
    :param i: index of the grape
    :param x: x coordinate of end left corner of the window
    :param y: y coordinate of end left corner of the window
    """
    if img is not None:
        target_bolded = img.copy()
        # if i is not None:
        #     # if not g_param.TB[i].sprayed:
        #     #     print("grape to display: ", g_param.TB[i])
        #     cv.drawContours(target_bolded, [np.asarray(g_param.TB[i].p_corners)], 0, (15, 25, 253), thickness=3)
        cv.namedWindow(win_name, cv.WINDOW_AUTOSIZE)  # Create a named window
        cv.moveWindow(win_name, x, y)  # Move it to (x,y)
        # cv.resizeWindow(win_name, 400, 512)
        target_bolded = cv.resize(target_bolded, (768, 768))  # TODO- comment if working with second screen
        cv.imshow(win_name, target_bolded)
        cv.waitKey()
        cv.destroyAllWindows()


def show_in_moved_window_double(win_name, img_1, img_2, i=None, x=0, y=0):  # lab
    """
    show image
    :param win_name: name of the window
    :param img: image to display
    :param i: index of the grape
    :param x: x coordinate of end left corner of the window
    :param y: y coordinate of end left corner of the window
    """
    if img_1 is not None:
        target_bolded = img_1.copy()
        # if i is not None:
        #     # if not g_param.TB[i].sprayed:
        #     #     print("grape to display: ", g_param.TB[i])
        #     cv.drawContours(target_bolded, [np.asarray(g_param.TB[i].p_corners)], 0, (15, 25, 253), thickness=3)
        cv.namedWindow(win_name, cv.WINDOW_AUTOSIZE)  # Create a named window
        cv.moveWindow(win_name, x, y)  # Move it to (x,y)
        # cv.resizeWindow(win_name, 400, 512)
        # target_bolded = cv.resize(target_bolded, (768, 768))  # TODO- comment if working with second screen
        # img_2 = cv.resize(target_bolded, (768, 768)) # TODO- comment if working with second screen
        len_img_2 = len([a for a in img_2.shape])
        if len_img_2 > 2:
            img_2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)
        target_bolded = cv.hconcat((target_bolded, img_2))
        target_bolded = cv.resize(target_bolded, (int(target_bolded.shape[1] * 0.7),
                                                  int(target_bolded.shape[0] * 0.7)))
        cv.imshow(win_name, target_bolded)
        cv.waitKey()
        cv.destroyAllWindows()


old_exp_path = r'C:\Users\Administrator\Desktop\Edo_dir\grapes\June exp data'
old_exps = os.listdir(old_exp_path)


def connected_dots(binary_map):
    # do connected components processing
    nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(binary_map, None, None, None, 8, cv.CV_32S)

    # get CC_STAT_AREA component as stats[label, COLUMN]
    areas = stats[1:, cv.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if areas[i] >= 20:  # keep
            result[labels == i + 1] = 255
    show_in_moved_window_double(win_name="Remove dots", img_1=binary_map, img_2=result, x=-1450, y=-200)
    return result


def canny(working_copy, gaussian, laplace, close_morpho, open_morpho):
    grayscale = cv.cvtColor(working_copy, cv.COLOR_RGB2GRAY)  # convert to grayscale
    threshValue = 140
    _, binaryImage = cv.threshold(grayscale, threshValue, 255, cv.THRESH_TOZERO_INV)  # remove white areas

    # apply gaussian blur before canny edge detection
    if gaussian:
        kernel_size = 3
        gaussian_blurred = cv.GaussianBlur(binaryImage, (kernel_size, kernel_size), 0)
        # show_in_moved_window_double(win_name="Gaussian", img_1=binaryImage, img_2=gaussian_blurred, x=-1450, y=-200)
        binaryImage = gaussian_blurred
    if laplace:
        laplacian = cv.Laplacian(binaryImage, cv.CV_64F, ksize=7)
        # show_in_moved_window("laplacian", laplacian)
        binaryImage = laplacian
    # low_threshold, high_threshold = 12, 35
    # edges = cv.Canny(gaussian_blurred, low_threshold, high_threshold)
    th, bw = cv.threshold(binaryImage, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # calc threshold for canny
    edges = cv.Canny(binaryImage, th / 2, th)
    # show_in_moved_window("canny", edges)
    edges = connected_dots(edges)

    kernel = np.ones((3, 3), np.uint8)
    if close_morpho:
        edges_1 = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
        # show_in_moved_window_double(win_name="morphology", img_1=edges, img_2=edges_1, x=-1450, y=-200)
        edges = edges_1
    if open_morpho:
        edges_2 = cv.morphologyEx(edges, cv.MORPH_OPEN, kernel)
        # show_in_moved_window_double(win_name="morphology", img_1=edges_4, img_2=edges_2, x=-1450, y=-200)
        edges = edges_2
    return edges, binaryImage

count = 0
for dir in old_exps:
    count += 1
    if count < 12:
        continue
    dir = os.path.join(old_exp_path, dir)
    dir_masks = os.path.join(dir, "masks")
    if len(os.listdir(dir_masks)) > 0:
        print(dir[-20:])
        masks_path = dir_masks
        img_path = os.path.join(dir, r"rgb_images\resized")
        images = os.listdir(img_path)
        npzs_without = os.listdir(masks_path)
        npzs_without = [x.split('_')[0] for x in npzs_without]
        npzs_without_int = [int(x.split('_')[0]) for x in npzs_without]
        images = [x for x in images if x.split('_')[0] in npzs_without]

        image_list = []
        npzs = os.listdir(masks_path)
        for i in range(len(npzs)):
            image_list.append([x for x in images if (x.split('_')[0] in npzs[i].split('_')[0] and len(x.split('_')[0]) == len(npzs[i].split('_')[0]))])
        image_list = sum(image_list, [])
        pairs = [list(x) for x in zip(npzs, image_list, npzs_without_int)]

        pairs = sorted(pairs, key=lambda x: x[2])
        print(pairs)

        for i in range(len(os.listdir(dir_masks))):
            print(len(os.listdir(dir_masks)))
            path = masks_path
            npz_path = os.path.join(path, pairs[i][0])
            mask_npz = np.load(npz_path)
            mask = mask_npz.f.arr_0
            path = img_path
            image_path = os.path.join(path, pairs[i][1])
            image = cv.imread(image_path)
            # image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
            redMask = cv.bitwise_and(image, image, mask=mask)
            # cv.addWeighted(redMask, 1, image, 1, 0, image)
            working_copy = redMask.copy()
            # show_in_moved_window('image', image)
            hsv = cv.cvtColor(working_copy, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, (30, 20, 20), (70, 255, 255))
            imask = mask > 0
            green = np.zeros_like(working_copy, np.uint8)
            green[imask] = working_copy[imask]
            show_in_moved_window("green", green)

            working_copy = green.copy()

            # Sobel Edge Detection
            # img_blur = working_copy
            # sobelx = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=0,
            #                    ksize=3)  # Sobel Edge Detection on the X axis
            # sobely = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=0, dy=1,
            #                    ksize=3)  # Sobel Edge Detection on the Y axis
            # sobelxy = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=1,
            #                     ksize=5)  # Combined X and Y Sobel Edge Detection
            # # Display Sobel Edge Detection Images
            # show_in_moved_window('Sobel X', sobelx)
            # show_in_moved_window('Sobel Y', sobely)
            # show_in_moved_window('Sobel XY', sobelxy)

            image_canny, thresholded_image = canny(working_copy=working_copy, gaussian=True, laplace=False,
                                                   open_morpho=False, close_morpho=False)
            show_in_moved_window_double(win_name=f'{pairs[i][2]}, after morpho, gussian, binary, canny',
                                        img_1=image_canny, img_2=green, i=None, x=-1450, y=-200)  # Laptop only
            # show_in_moved_window(win_name=f'{pairs[i][2]}', img=image_canny, i=None, x=0, y=0)  # Laptop only
            # show_in_moved_window(win_name=f'{pairs[i][2]}', img=redMask, i=None, x=0, y=0) # Laptop only
            # show_in_moved_window(win_name=f'{pairs[i][2]}',
            # img=image, i=None, x=-1250, y=-300) # Second monitor on the left







view_from_windshield = image.imread('circels.jpg')
working_copy = np.copy(view_from_windshield)

# first convert to grayscale

#pyplot.imshow(grayscale, cmap='gray')
#pyplot.imshow(gaussian_blurred, cmap='gray')
# pyplot.imshow(edges, cmap='Greys_r')
# pyplot.show()
