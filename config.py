import numpy as np


class Config:
    def __init__(self):
        # TODO: when working on my computer. change to 200, 200 when working in lab
        self.screen_x_cord = -1_350
        self.screen_y_cord = -270
        self.show_obb = False
        self.work_type = "not defined yet"
        self.menu_new = "Enter 1 for annotate new image/re-annotate image \n"\
                        f"Enter 2 to change work type- current is {self.work_type} \n"\
                        "Enter 3 to exit"
        self.menu_fix = "Enter 1 for display old masks \n"\
                        "Enter 2 for adding masks to annotated image \n" \
                        "Enter 3 for annotating new image \n" \
                        f"Enter 4 to change work type- current is {self.work_type} \n"\
                        "Enter 5 to exit \n"\
                        "Enter 6 to delete masks"

        self.clean_folder = 3
        self.new_mask = False
        # self.const_part_img_name = 'DSC_0'
        # self.x_input_dim = 4000
        # self.y_input_dim = 6000
        self.const_part_img_name = ''
        # self.x_input_dim = 1024
        # self.y_input_dim = 1024
        self.input_image_dim = np.asarray([1024, 1024, 3])
        # TODO: change resize_image parameters in utils.

    def print_menu(self):
        if self.work_type == "new":
            print(self.menu_new)
        else:
            print(self.menu_fix)

config = Config()
