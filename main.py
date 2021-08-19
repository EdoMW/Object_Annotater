import os
import shutil

import cv2
import scipy

import utils
import math
from random import randint
import matplotlib.pyplot as plt
import gc
from config import *

screen_x_cord, screen_y_cord = config.screen_x_cord, config.screen_y_cord
show_obb = config.show_obb


def read_image(img_path_to_read):
    """
    read 6000/4000/3 rgb image. resize it. return parameters to later resize the mask to the original size.
    :return: image_resized, scale_ratio, padding_dim
    """
    try:
        image = cv2.imread(img_path_to_read)
        config.input_image_dim = np.asarray(image.shape)
        # self.x_input_dim = config.input_image_dim[0]
        # self.y_input_dim = 1024

        image_resized, _, scale_ratio, padding_dim, _ = utils.resize_image(image, 1024, 1024)
        return image_resized, scale_ratio, padding_dim
    except:
        print(f"Wrong format of image {img_path_to_read.split('.')[1]}. insert JPG images only.")


def show_img(title, image_to_show, x_cord, y_cord):
    cv2.imshow(title, image_to_show)
    cv2.moveWindow(title, x_cord, y_cord)


def get_rect_box(cnt):
    hull = cv2.convexHull(cnt)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return rect, box


class DrawLineWidget(object):
    """
    The algorithm that enable draw new masks and calculate OBB's.
    Mouse clicks:
    left button click- add new point. each click added (after the first one) will draw a line between them.
    if click was done close enough to the first (red) point, the mask will be saved. at least 3 points are required.
    right button click- clear the last mask that was in process.
    toggle scroll - erase the last point that was added (line/lines to be deleted marked in blue).
    Enter/Esc button- close the polygon with straight line between the first and last points.
    """

    def __init__(self, img_shared):
        self.original_image = img_shared
        self.clone = img_shared.copy()
        self.clone_2 = self.clone.copy()
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.extract_coordinates)
        self.image_coordinates = []

    def extract_coordinates(self, event, x, y, *args):
        # Record starting (x,y) coordinates on left mouse button click (next two if, elif)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clone_2 = self.clone.copy()
            self.image_coordinates.append((x, y))
            x_start, y_start = self.image_coordinates[0][0], self.image_coordinates[0][1]
            if len(self.image_coordinates) > 1:
                if abs(((x - x_start) ** 2) + ((y - y_start) ** 2)) < 95:
                    # print("finished")
                    cv2.destroyAllWindows()
        elif event == cv2.EVENT_LBUTTONUP:
            # self.image_coordinates.append((x,y))
            len_line = len(self.image_coordinates)
            # print(f'Amount of verices: {len_line}', "mouse up", self.image_coordinates)
            self.clone = self.original_image.copy()
            if len_line > 0:
                cv2.circle(self.clone, (int(self.image_coordinates[0][0]), int(self.image_coordinates[0][1])),
                           radius=3, color=(12, 36, 255), thickness=3)
                cv2.circle(self.clone, (int(self.image_coordinates[-1][0]), int(self.image_coordinates[-1][1])),
                           radius=2, color=(36, 255, 12), thickness=3)
            if len_line > 1:
                for j in range(len(self.image_coordinates) - 1):
                    cv2.line(self.clone, self.image_coordinates[j], self.image_coordinates[j + 1], (36, 255, 12), 2)
                # cv2.line(self.clone, self.image_coordinates[len_line-2],
                # self.image_coordinates[len_line-1], (36,255,12), 2)
            show_img(title="image", image_to_show=self.clone, x_cord=screen_x_cord, y_cord=screen_y_cord)
            # self.clone_2 = self.clone.copy()

        # Clear drawing boxes on right mouse button click (next two elif)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.original_image.copy()
            self.image_coordinates = []
        elif event == cv2.EVENT_RBUTTONUP:
            # cv2.imshow("image", self.clone)
            show_img(title="image", image_to_show=self.clone, x_cord=screen_x_cord, y_cord=screen_y_cord)

        # Clear last point added on mouse wheel scroll
        elif event == cv2.EVENT_MOUSEWHEEL:
            cv2.line(self.clone_2, self.image_coordinates[-2], self.image_coordinates[-1], (255, 36, 12), 2)
            self.image_coordinates.pop()
            # print(self.image_coordinates)
            # cv2.imshow("image", self.clone_2)
            show_img(title="image", image_to_show=self.clone_2, x_cord=screen_x_cord, y_cord=screen_y_cord)
            # self.clone_2 = self.clone.copy()
        elif event == cv2.EVENT_MOUSEMOVE:
            if len(self.image_coordinates) > 2:
                x_start, y_start = self.image_coordinates[0][0], self.image_coordinates[0][1]
                if abs(((x - x_start) ** 2) + ((y - y_start) ** 2)) < 175:
                    cv2.circle(self.clone, (int(self.image_coordinates[0][0]), int(self.image_coordinates[0][1])),
                               radius=7, color=(120, 36, 255), thickness=3)
            pass

    def show_image(self):
        return self.clone


def get_unrecognized_grapes(img_t, amount_of_grapes):
    """
    Let you draw new grapes that were not annotated.
    :param img_t: image with current grape masks that were detected.
    :param amount_of_grapes: amount of new grapes to be added
    :return: obbs, corners, npys, img_shared
    """
    img_shared = img_t.copy()
    obbs, npys, corners = [], [], []
    image_index = 0
    # amount_of_grapes = int(input("enter amount of grapes: "))
    # amount_of_grapes = 2
    while image_index < amount_of_grapes:
        draw_line_widget = DrawLineWidget(img_shared)
        while True:
            show_img(title='image', image_to_show=draw_line_widget.show_image(),
                     x_cord=screen_x_cord, y_cord=screen_y_cord)
            if cv2.waitKey() or 0xFF == 27:
                break
        cv2.destroyAllWindows()
        cv2.waitKey(0) & 0xFF
        array = draw_line_widget.image_coordinates  # no predefined polygon
        if len(array) < 3:
            continue
        image_index += 1
        print(f'{image_index}/{amount_of_grapes} masks were added so far.')
        img = img_t.copy()
        for i in range(len(array) - 1):
            cv2.line(img, array[i], array[i + 1], (36, 255, 12), 2)
        cv2.line(img, array[0], array[i + 1], (36, 255, 12), 2)
        cv2.destroyAllWindows()
        array = [list(ele) for ele in array]
        array = np.array([np.array(xi) for xi in array])
        points = array
        for i in range(len(points)):
            cv2.circle(img, (int(points[i][0]), int(points[i][1])), radius=4, color=(12, 12, 255),
                       thickness=3)
            # cv2.putText(img, f'{(int(points[i][0]), int(points[i][1]))}',
            #             org=(int(points[i][0]) - 75, int(points[i][1]) + 35),
            #             fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.51,
            #             color=(255, 255, 255), thickness=1, lineType=2)
        img_1 = np.zeros([img.shape[0], img.shape[1], 3], dtype=np.uint8)
        img_1.fill(255)

        cv2.polylines(img_1, [array], True, (0, 15, 255))
        cv2.drawContours(img_1, [array], -1, (0, 15, 255), -1)
        src_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(src_gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        rect, box = get_rect_box(contours[0])
        npy = np.asarray(thresh, dtype=np.uint8)
        if show_obb:
            img_shared = cv2.drawContours(img_shared, [box], 0, (10, 154, 255), 2)
        # if image_index == amount_of_grapes: # optional - indent lines 141-151 and add the if.
        img_shared_overlay = img_shared.copy()
        cv2.polylines(img_shared_overlay, [array], True, (0, 255, 255))
        cv2.fillPoly(img_shared_overlay, [array], 255)
        alpha = 0.4
        cv2.addWeighted(img_shared_overlay, alpha, img_shared, 1 - alpha,
                        0, img_shared)
        show_img(title=f"image number {image_index + 1}", image_to_show=img_shared,
                 x_cord=screen_x_cord, y_cord=screen_y_cord)
        # cv2.imshow()
        cv2.waitKey()
        cv2.destroyAllWindows()
        obbs.append(rect)
        corners.append(box)
        npys.append(npy)

    return obbs, corners, npys, img_shared


def display_image_with_mask(image_path_to_display, mask_path_to_display, mask_npz_path=None, display=1, alpha=0.7):
    """
    display: 1 for display, 2 for delete
    Iterate through mask file and overlay
    """
    # print(image_path_to_display)
    image = cv2.imread(image_path_to_display)
    if mask_npz_path is not None:
        try:
            a = np.load(mask_path_npz)
            mask = a.f.arr_0
            np.save(file=mask_path_to_display, arr=mask)
        except FileNotFoundError:
            # print(f"No exiting annotation yet for image {image_path_to_display[-7:-4]}")
            return
    else:
        mask = np.load(mask_path_to_display)

    im = image.copy()
    # print("Image shape: ", image.shape)
    # print("Mask shape: ", mask.shape)
    for i in range(len(mask[0][0] + 1)):
        r, g, b = randint(0, 255), randint(0, 255), randint(0, 255)
        rgb = (r, g, b)
        temp_mask = mask[:, :, i]
        mask_temp = temp_mask.copy()
        image[mask_temp == 1] = rgb
        if display == 2:
            x_center, y_center = calc_center_of_mass(temp_mask)
            image = cv2.putText(image, str(i), org=(int(y_center), int(x_center)),
                                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=5,
                                color=(255, 255, 255), thickness=5, lineType=2)
    plt.figure(figsize=(12, 8))
    plt.title(f'{image_path_to_display[:-4]}', fontweight="bold")
    plt.imshow(im)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image, 'gray', interpolation='none', alpha=alpha, vmin=1)  # alpha: 0-1, 0 is 100% transparency, 1 is 0%
    plt.show()
    f_name = image_path_to_display[:-4] + '_masked.jpg'
    plt.imsave(fname=f_name, arr=image)
    return (mask.shape[2])


def calc_center_of_mass(m):
    """
    calc mask center of mass
    :param m: mask
    :return: center of mask (x,y)
    """
    if m is not None:
        return scipy.ndimage.measurements.center_of_mass(m)
    return None


def fix_mask_size(mask):
    x_mask, y_mask, z_mask = mask.shape[0], mask.shape[1], mask.shape[2]
    x_orig, y_orig, z_orig = config.input_image_dim[0], config.input_image_dim[1], z_mask  # config.input_image_dim[2]
    x_offset, y_offset = 0, 0
    if x_mask == x_orig and y_mask == y_orig:
        return mask
    else:
        result = np.zeros((x_orig, y_orig, z_orig))
        if x_mask != x_orig:
            dif_x = abs(x_mask - x_orig)
            if x_mask > x_orig:
                mask = mask[math.floor(dif_x / 2): -math.ceil(dif_x / 2), :, :]
            else:
                x_offset = math.floor(dif_x / 2)
                result[x_offset:mask.shape[0] + x_offset, y_offset:mask.shape[1] + y_offset, :] = mask
                mask = result
        if y_mask != y_orig:
            dif_y = y_mask - y_orig
            if y_mask > y_orig:
                mask = mask[:, math.floor(dif_y / 2): -math.ceil(dif_y / 2), :]
            else:
                y_offset = math.floor(dif_y / 2)
                result[x_offset:mask.shape[0] + x_offset, y_offset:mask.shape[1] + y_offset, :] = mask
                mask = result
    return mask


def annotate_image(new_mask_ann=True):
    """
    main function for annotating new image
    :param new_mask_ann: True if it's an annotation of unannotated or poorly annotated image.
    """
    obbs_list, corners_list, npys_list = [], [], []
    if not new_mask_ann:
        img_path = image_path[:-4] + '_masked.jpg'
        mask_path_1 = 'npzs/' + str(mask_path[5:])
        a = np.load(mask_path_1[:-4] + '.npz')
        npy_orig = a.f.arr_0
        # npy_orig = np.load(mask_path)
    else:
        img_path = image_path
    img_rgb, scale, padding = read_image(img_path)
    first_round = True
    while True:
        if first_round:
            check_for_more = '1'
            first_round = False
        else:
            check_for_more = input("\n More grapes to add?" '\n'
                                   "if None press ENTER to continue. Else enter positive number: ")
        if check_for_more == "":
            break
        elif check_for_more.isdigit():
            obbs_temp, corners_list_temp, npys_temp, img_rgb = get_unrecognized_grapes(img_rgb, int(check_for_more))
            for grapes_added in range(int(check_for_more)):
                obbs_list.append(obbs_temp[grapes_added])
                corners_list.append(corners_list_temp[grapes_added])
                npys_list.append(npys_temp[grapes_added])
    print(f'total of {len(npys_list)} grapes were added manually')
    # if there were grapes to be added to the Data Base.
    if len(npys_list) or len(npy_orig.shape[2]) > 0:
        # converting the array to np array in shape of [1024, 1024, N]
        npy_array = npys_list[0]
        if len(npys_list) > 1:
            for arr in range(1, len(npys_list)):
                npy_array = np.dstack((npy_array, npys_list[arr]))
        else:  # 1024,1024 is for squre image in this dim. can be changed.
            npy_array = np.reshape(npy_array, (1024, 1024, 1))
        # npy_array = npy_array[170:853, :, :]  # remove the padding
        mask = utils.resize_mask(npy_array, 1 / scale, padding=0, crop=None)  # make it [4002,6000,N]
        mask = fix_mask_size(mask)
        # mask = mask[1:4001, :, :]  # fix the resizing
        mask = mask.clip(max=1)  # fix colors in visualization module
        if not new_mask_ann:
            mask = np.dstack((mask, npy_orig))
        np.save(file=mask_path, arr=mask)  # save the name mask.
        np.savez_compressed(mask_path_npz, mask)


def fix_mask(image_path, mask_path, mask_path_npz):
    """
    fix old mask. first load npz file and display the old masks annotation on top of the image.
    continue to annotate new masks.
    """
    _ = display_image_with_mask(image_path_to_display=image_path, mask_path_to_display=mask_path,
                                mask_npz_path=mask_path_npz, display=1)
    annotate_image(config.new_mask)


def annotate_new_image(is_new_mask):
    is_new_mask = not is_new_mask
    annotate_image(is_new_mask)


def print_information(images_list_to_display, npz_l_images):
    """
    Print information about the images
    :param images_list_to_display: images dir
    :param npz_l_images: npzs dir
    """
    # images_len = len([x for x in os.listdir('images') if x[-7:-4] != 'ked'])
    images_len = len([x for x in os.listdir('images') if x[-7:-4] != 'ked'])
    if len(images_list_to_display) > 0:
        if config.work_type == "new":
            print(f'Following images have not been annotated yet {images_list_to_display[:4]}, '
                  f'total of {len(images_list_to_display)} remained out of {images_len} images]')
        else:
            print(f'Following images have been annotated {images_list_to_display[:4]}, '
                  f'total of {len(images_list_to_display)} select image to fix/add annotations]')
    else:
        print("All images have been annotated!")
        answer = input("Enter 5 to view list of annotated images, Enter otherwise: \n")
        if answer.isdigit():
            if int(answer) == 5:
                # npz_l_images = [x[-7:-4] for x in npz_l_images]
                npz_l_images = [x[:-4] for x in npz_l_images]
                print("npz_l_images: ", npz_l_images)
                return True
    return False


def choose_work_type():
    start = "\033[1m"
    end = "\033[0;0m"
    print("Press 1 to annotate new images \nPress 2 to add/fix annotations")
    while True:
        work = str(input(""))
        if work.isdigit():
            if work == '1':
                config.work_type = "new"
                break
            elif work == '2':
                config.work_type = "fix"
                break
            else:
                print("Please choose an option")
        else:
            print("Please choose valid option")


def print_next_images():
    """
    print information about the images
    """
    images_list = os.listdir('images')
    npz_list = os.listdir('npzs')
    npz_list_images = [x[:-4] + '.jpg' for x in npz_list]
    if config.work_type == "new":
        images_list = [x for x in images_list if x not in npz_list_images and x[-7:-4] != 'ked']
        # images_list = [x[-7:-4] for x in images_list if x[-7:-4] != 'ked']
        images_list = [x[:-4] for x in images_list if x[:-4] != 'ked']
    else:
        # images_list = [x[-7:-4] for x in images_list if x in npz_list_images]
        images_list = [x[:-4] for x in images_list if x in npz_list_images]

    allow_choice = print_information(images_list, npz_list_images)
    return images_list[0] if len(images_list) > 0 else None, allow_choice


def select_image(im_num, pick_number):
    """
    Enter image number
    :return: image number
    """
    option_1 = None
    const_img_name = config.const_part_img_name
    if img_num is None and not pick_number:
        return "skip"
    while True:
        option_1 = const_img_name + str(input(f"{im_num} selected. Press enter to confirm"
                                              f" or insert 3 digits to select other image: "))
        # if option_1[-3:].isdecimal():
        if option_1 is not None:
            # to verify fiverr images
            # if all([x for x in option_1.split('_') if x.isdecimal]) and len(option_1.split('_')) == 4:
            if option_1 != "":
                break
            elif option_1[len(const_img_name):] == "":
                option_1 += im_num
                break
    return option_1


def choose_option():
    """
    Pick an option from the menu
    :return: option
    """
    config.print_menu()
    choose_number = None
    while True:
        option = input("")
        if option.isdecimal():
            choose_number = int(option)
            break
    if config.work_type == 'new':
        choose_number += 2
    return choose_number


def clean_npy_dir():
    """
    Empty the npy_dir at the end of the program or when 3 (clean_folder) images in a row were annotated. (save space)
    """
    shutil.rmtree('npys')
    os.mkdir('npys')


def remove_masked_images():
    """
    remove masked images without npz files
    :return:
    """
    npz_to_keep = [x for x in os.listdir('images') if x[-7:-4] == 'ked']
    npzs = [x[:-4] + '.jpg' for x in os.listdir('npzs')]

    np_t_k = [x[:8] + '.jpg' for x in npz_to_keep]
    intersect = list(set.intersection(set(np_t_k), set(npzs)))
    intersect_masked = [x[:8] + '_masked.jpg' for x in intersect]
    for i in range(len(npz_to_keep)):
        if npz_to_keep[i] not in intersect_masked:
            os.remove(f'images/{npz_to_keep[i]}')


def init_folders():
    """
    in case it's the first iteration
    :return:
    """
    end_prog = False
    if not os.path.exists('images'):
        os.mkdir('images')
        print("Move images that you want to annotate, to the new directory 'images'")
        end_prog = True
    elif len(os.listdir('images')) == 0:
        print("Images directory is empty. insert images to annotate")
        end_prog = True
    if not os.path.exists('npys'): os.mkdir('npys')
    if not os.path.exists('npzs'): os.mkdir('npzs')
    remove_masked_images()
    return end_prog


def convert_annotations():
    """
    TODO: convert annotations to different formats.
    """
    pass


def make_paths(img_name):
    """
    :param img_name: image number
    :return: paths to image, mask and mask path.
    """
    image_p = 'images/' + img_name + '.JPG'
    mask_p = 'npys/' + img_name + '.npy'
    mask_path_n = 'npzs/' + img_name + '.npz'
    return image_p, mask_p, mask_path_n


def change_work_type():
    """
    Change work type (from annotate new image to fix, or from fix to annotate new image.
    """
    if config.work_type == 'new':
        config.work_type = 'fix'
    elif config.work_type == 'fix':
        config.work_type = 'new'


def delete_masks(image_path, mask_path, mask_path_npz):
    masks_count = display_image_with_mask(image_path_to_display=image_path, mask_path_to_display=mask_path,
                                          mask_npz_path=mask_path_npz, display=2)
    empty_file = False
    while True:
        if masks_count == 0:
            print("No masks to delete")
            if mask_path_npz is not None:
                empty_file = True
            break
        mask_to_rem = str(input("Please enter mask number to delete"))
        if mask_to_rem.isdigit():
            if int(0 <= int(mask_to_rem) < masks_count):
                a = np.load(mask_path_npz)
                mask = a.f.arr_0
                print(mask.shape)
                mask = np.dstack((mask[:, :, :int(mask_to_rem)], mask[:, :, int(mask_to_rem) + 1:]))
                # mask = np.delete(mask, mask[:, :, int(mask_to_rem)])
                print(mask.shape)
                print(mask_path_npz)
                if mask.shape[2] > 0:
                    np.savez_compressed(mask_path_npz, mask)
                else:
                    masks_count = 0
                _ = display_image_with_mask(image_path_to_display=image_path, mask_path_to_display=mask_path,
                                            mask_npz_path=mask_path_npz, display=2)
            else:
                print(f"Please enter number between 0 to {masks_count} to delete the mask")
        elif mask_to_rem == "":
            print("Exit, no more masks delete.")
            break
        else:
            print("Please enter valid mask number, or press enter to exit")
    return empty_file


def del_masks(*args):
    del_file = delete_masks(*args)
    if del_file:
        os.remove(mask_path_npz)


def take_action(option, image_path, mask_path, mask_path_npz, config):
    if option == 1:
        display_image_with_mask(image_path_to_display=image_path,
                                mask_path_to_display=mask_path,
                                mask_npz_path=mask_path_npz)
    elif option == 2:
        fix_mask(image_path, mask_path, mask_path_npz)
    elif option == 3:
        annotate_new_image(config.new_mask)
    elif option == 4:
        change_work_type()
    elif option == 5:
        return True
    elif option == 6:
        del_masks(image_path, mask_path, mask_path_npz)
    return False


if __name__ == "__main__":
    """
    change "name" if it's not from the Nikon camera
    """
    end_program = init_folders()
    choose_work_type()
    image_path, mask_path, mask_path_npz = None, None, None
    while True and not end_program:
        gc.collect()
        img_num, pick_num = print_next_images()
        name = select_image(img_num, pick_num)
        if name == 'skip':
            change_work_type()
            # continue
        else:
            image_path, mask_path, mask_path_npz = make_paths(name)
        option = choose_option()
        end_program = take_action(option, image_path, mask_path, mask_path_npz, config)
        if len(os.listdir(r'npys')) > config.clean_folder:
            clean_npy_dir()
    convert_annotations()
    clean_npy_dir()
