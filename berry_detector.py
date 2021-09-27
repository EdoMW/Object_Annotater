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
    widths = stats[1:, cv.CC_STAT_WIDTH]
    heights = stats[1:, cv.CC_STAT_HEIGHT]
    result = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if areas[i] >= 30:
        # if areas[i] >= 30 and 0.33 < widths[i] / heights[i] < 3 and 200 < widths[i]*heights[i]:  # filter
            result[labels == i + 1] = 255

    result = cv.dilate(result, kernel=np.ones((2, 2), np.uint8), iterations=1)  # make the lines thicker
    show_in_moved_window_double(win_name="Remove dots", img_1=binary_map, img_2=result, x=-1450, y=-200)
    plt.figure()
    plt.axis("on")
    plt.imshow(result)
    plt.show()
    return result


def adaptive_threshold(image_guss, img_color):
    thresh = cv.adaptiveThreshold(image_guss, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 11, 2)
    contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour_list = []
    for contour in contours:
        approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
        area = cv.contourArea(contour)
        # Filter based on length and area
        print(len(approx), area)
        if (2 < len(approx) < 18) & (900 > area > 20):
            # print area
            contour_list.append(contour)

    cv.drawContours(img_color, contour_list, -1, (255, 20, 20), 2)
    cv.imshow('Objects Detected', img_color)
    cv.waitKey(5000)


def circles(image_guss, img_color):
    circles = cv.HoughCircles(image_guss, cv.HOUGH_GRADIENT, 1, minDist=30,
                               param1=40, param2=20, minRadius=0, maxRadius=50)
    if circles is not None:
        for i in circles[0, :]:
            # draw the outer circle
            cv.circle(img_color, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)
            # draw the center of the circle
            # cv.circle(img_color, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv.imshow('circles', img_color)
    cv.waitKey(5000)


def show_hist(img):
    histogram, bin_edges = np.histogram(img, bins=256, range=(0, 1))
    hist = cv.calcHist([img], [0], None, [256], [1, 256])
    plt.figure()
    plt.axis("off")
    plt.imshow(cv.cvtColor(img, cv.COLOR_GRAY2RGB))
    # img *= 2 #FIXME
    # cv.imshow("ggg", img)
    # cv.waitKey()
    # plot the histogram
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(hist)
    plt.xlim([0, 256])

    # plt.show()
    # plot the normalized histogram
    # hist /= hist.sum()
    # plt.figure()
    # plt.title("Grayscale Histogram (Normalized)")
    # plt.xlabel("Bins")
    # plt.ylabel("% of Pixels")
    # plt.plot(hist)
    # plt.xlim([0, 256])
    # plt.show()
    # show_in_moved_window(win_name="Gray", img=histogram, x=-1450, y=-200)

    img_bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    img_yuv = cv.cvtColor(img_bgr, cv.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv.equalizeHist(img_yuv[:, :, 0])
    # convert the YUV image back to RGB format
    img_output = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)
    # show_in_moved_window('Color input image', img_bgr)
    # show_in_moved_window('Histogram equalized', img_output)
    # show_in_moved_window('Histogram equalized', img_output)
    img_output_bgr = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)
    img_output_gray = cv.cvtColor(img_output_bgr, cv.COLOR_BGR2GRAY)
    # show_in_moved_window('Histogram equalized', img_output_gray)
    return img_output_gray
    # return img


def canny(img_color, working_copy, gaussian, laplace, close_morpho, open_morpho):
    grayscale = cv.cvtColor(working_copy, cv.COLOR_RGB2GRAY)  # convert to grayscale
    threshValue = 140
    _, binaryImage = cv.threshold(grayscale, threshValue, 255, cv.THRESH_TOZERO_INV)  # remove white areas
    # show_in_moved_window(win_name="Gray", img=binaryImage, x=-1450, y=-200)
    binaryImage = show_hist(binaryImage)
    # apply gaussian blur before canny edge detection
    if gaussian:
        kernel_size = 3
        gaussian_blurred = cv.GaussianBlur(binaryImage, (kernel_size, kernel_size), 0)
        # show_in_moved_window_double(win_name="Gaussian", img_1=binaryImage, img_2=gaussian_blurred, x=-1450, y=-200)
        binaryImage_2 = gaussian_blurred.copy()
        binaryImage = gaussian_blurred.copy()
        # gaussian_blurred_2 = cv.GaussianBlur(binaryImage_2, (kernel_size*3, kernel_size*3), 0)
        # show_in_moved_window_double(win_name="Gaussian2", img_1=binaryImage_2, img_2=gaussian_blurred_2, x=-1450, y=-200)
        # binaryImage = gaussian_blurred
    if laplace:
        laplacian = cv.Laplacian(binaryImage, cv.CV_64F, ksize=7)
        show_in_moved_window("laplacian", laplacian)
        binaryImage = laplacian
    binaryImage = binaryImage.astype('uint8')
    # low_threshold, high_threshold = 50, 125
    # edges = cv.Canny(binaryImage, low_threshold, high_threshold)
    # show_in_moved_window("binaryImage before canny", binaryImage)
    th, bw = cv.threshold(binaryImage, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # calc threshold for canny
    edges = cv.Canny(binaryImage, th*0.7, th*1.2)
    # show_in_moved_window("canny", edges)
    edges = connected_dots(edges)

    # rgb_double = cv.cvtColor(binaryImage, cv.COLOR_GRAY2RGB)  # FIXME
    # circles(binaryImage, rgb_double) # FIXME
    # adaptive_threshold(binaryImage, img_color)  # not detecting anything..




    kernel = np.ones((3, 3), np.uint8)
    if close_morpho:
        edges_1 = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
        # show_in_moved_window_double(win_name="morphology", img_1=edges, img_2=edges_1, x=-1450, y=-200)
        edges = edges_1
    if open_morpho:
        edges_2 = cv.morphologyEx(edges, cv.MORPH_OPEN, kernel)
        # show_in_moved_window_double(win_name="morphology", img_1=edges_4, img_2=edges_2, x=-1450, y=-200)
        edges = edges_2

    from skimage.transform import hough_ellipse
    from skimage.draw import ellipse_perimeter
    from skimage import data, color, img_as_ubyte

    return edges, binaryImage

count = 0
for dir in old_exps:
    count += 1
    if count < 15:
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
            # show_in_moved_window("green", green)

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

            image_canny, filterd_binary_img = canny(img_color=green, working_copy=working_copy, gaussian=True, laplace=False,
                                                   open_morpho=False, close_morpho=False)
            show_in_moved_window_double(win_name=f'{pairs[i][2]}, after morpho, gussian, binary, canny',
                                        img_1=image_canny, img_2=filterd_binary_img, i=None, x=-1450, y=-200)  # Laptop only
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
