import sys
from mss import mss
import time
import numpy as np
from PIL import Image
from pymouse import PyMouse
from pykeyboard import PyKeyboard
import os
from pynput.keyboard import Key, Listener
import pickle

sys.setrecursionlimit(2000)

# note because screen array is inverted on the y axis angles start at the bottom and travel counter clockwise


class slither_bot():

    def __init__(self,
                 degrees_per_ray=5,
                 ray_starting_distance=30,
                 ray_length=300,
                 ray_point_space=10,
                 center_point={"x": 640, "y": 436},
                 colour_threshhold=200,  # minimum sum of rbg to classify as object
                 maximum_turn_per_frame=12,  # maximum degree adjustment per cycle
                 frame_capture_rate=0.5,  # 0 for every frame and None to not use
                 debug=False,
                 predator_min_size=300
                 ):

        if 90 % degrees_per_ray != 0:
            raise Exception("degrees_per_ray must be a factor of 90 but is " + degrees_per_ray)

        self.degrees_per_ray = degrees_per_ray
        self.ray_starting_distance = ray_starting_distance
        self.ray_length = ray_length
        self.ray_point_space = ray_point_space
        self.center_point = center_point  # point of slither head
        self.colour_threshhold = colour_threshhold
        self.maximum_turn_per_frame = maximum_turn_per_frame
        self.frame_capture_rate = frame_capture_rate
        self.debug = debug
        self.predator_min_size = predator_min_size

        self.monitor = {'top': 0, 'left': 0, 'width': 1280, 'height': 800}
        self.save_counter = 0

        self.mouse = PyMouse()
        self.keyboard = PyKeyboard()
        self.sct = mss()

    @staticmethod
    def open_img(path):
        """opens image and converts to rgb array"""

        im = Image.open(path, 'r')
        arr = np.asarray(im)
        arr.setflags(write=1)
        return arr

    def get_rays(self, screen_array):
        """returns list of rays where ray points are 1 if total colour is > threshold otherwise 0"""

        degrees_per_ray = self.degrees_per_ray
        ray_starting_distance = self.ray_starting_distance  # px
        ray_length = self.ray_length  # px
        ray_point_space = self.ray_point_space  # px
        center_point = self.center_point  # point of slither head
        rays_per_quadrant = int(90 / degrees_per_ray)

        if self.debug:
            arr = np.ones((int(360 / degrees_per_ray), int((ray_length - ray_starting_distance) / ray_point_space)), 'uint8')
            #arr = arr * 2
            return arr

        ray_list = []

        for quadrant in range(4):
            for ray in range(rays_per_quadrant):
                ray_list.append([])
                for ray_point in range(int((ray_length - ray_starting_distance) / ray_point_space)):

                    ray_actual = rays_per_quadrant - ray if (quadrant in [1, 3]) else ray
                    point_distance = ray_point * ray_point_space + ray_starting_distance

                    # get coordinates of ray point
                    x = int(np.sin(ray_actual * degrees_per_ray * np.pi / 180) * point_distance)
                    y = int(np.cos(ray_actual * degrees_per_ray * np.pi / 180) * point_distance)

                    if quadrant in [2, 3]:
                        x = -x
                    if quadrant in [1, 2]:
                        y = -y

                    # adjusts coordinates to run from center
                    x += center_point["x"]
                    y += center_point["y"]

                    # adds rgb values and compares to threshold
                    if sum(screen_array[y][x]) > self.colour_threshhold:
                        ray_list[quadrant * rays_per_quadrant + ray].append(1)
                    else:
                        ray_list[quadrant * rays_per_quadrant + ray].append(0)

        return ray_list

    def food_or_preditor(self, ray_list):
        """takes ray list and classifies mass as food 2 or predator 3 based on cluster size """

        for position_x in range(len(ray_list[0])):
            for position_y in range(len(ray_list)):
                if ray_list[position_y][position_x] == 1:
                    ray_list[position_y][position_x] = 4  # represents clustered but not classified

                    def find_all_ones_in_contact(pos_x, pos_y):
                        """recursively finds all mass that is in contact with starting point and returns list of coordinates"""

                        check_potitions = [[0, -1], [0, 1], [1, 0], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]
                        coordinates_list = []

                        for pos in check_potitions:

                            if len(ray_list[0]) > pos_x + pos[0] and len(ray_list) > pos_y + pos[1]:

                                # simulates an array curved around to meet its self
                                # this allows for finding clusters existing in +0 and -360 degree space
                                if pos_y + pos[1] < 0:
                                    pos[1] = 71 + pos[1]
                                if pos_x + pos[0] >= len(ray_list[0]) or pos_x + pos[0] < 0:
                                    continue

                                # checks if position has mass
                                if ray_list[pos_y + pos[1]][pos_x + pos[0]] == 1:

                                    ray_list[pos_y + pos[1]][pos_x + pos[0]] = 4  # is used to indicate that it has been clusstered but not assigned as 2 or 3
                                    output = find_all_ones_in_contact(pos_x + pos[0], pos_y + pos[1])  # runs function from current position

                                    if output != []:  # if function found more mass points they are added
                                        coordinates_list += output
                                    coordinates_list += [[pos_x + pos[0], pos_y + pos[1]]]

                        else:
                            return coordinates_list

                    list_positions_of_ones_in_cluster = find_all_ones_in_contact(position_x, position_y) + [[position_x, position_y]]

                    cluster_distance_from_center = np.amin(list_positions_of_ones_in_cluster, axis=0)[0]

                    total_size_score = 0
                    for pos in list_positions_of_ones_in_cluster:

                        degrees = np.float64(360 / len(ray_list))
                        disance_from_center = np.float64(self.ray_starting_distance + (self.ray_point_space * pos[0]))
                        distance_between_points = np.tan(degrees * np.pi / 180) * disance_from_center
                        total_size_score += distance_between_points

                    cluster_distance_from_center = np.amin(list_positions_of_ones_in_cluster, axis=0)[0]

                    if total_size_score > self.predator_min_size:
                        if cluster_distance_from_center == 0:
                            kind = 6  # assumes that this is its own tail
                        else:
                            kind = 3  # part of predator
                    else:
                        kind = 2

                    for i in list_positions_of_ones_in_cluster:
                        ray_list[i[1]][i[0]] = kind

        return ray_list

    def remove_food_on_sides(self, ray_list, current_direction):
        """sets food that is out of turning range to ignore value"""

        ray_current_direction = int(current_direction / 360) * (len(ray_list) - 1)

        food_remove_section = [int(ray_current_direction + len(ray_list) / 15), int(ray_current_direction - len(ray_list) / 15)]

        if food_remove_section[1] < 0:
            food_remove_section[1] += len(ray_list) - 1
        if food_remove_section[0] > len(ray_list) - 1:
            food_remove_section[0] += -(len(ray_list) - 1)

        list_of_rays_to_change = []

        i = food_remove_section[0]

        while True:
            if i > len(ray_list) - 1:
                i += -(len(ray_list))
            if i == food_remove_section[1]:
                break
            list_of_rays_to_change.append(i)

            i += 1

        for i in range(10):
            for j in list_of_rays_to_change:
                if ray_list[j][i] == 2:
                    ray_list[j][i] = 6
            list_of_rays_to_change = list_of_rays_to_change[1:-1]

        return ray_list

    def capture_frame(self, save_path=None):
        """takes screen shot and returns rgb array, can also save"""
        if self.debug:
            img = np.zeros((1280, 800, 3))
        else:
            img = self.sct.grab(self.monitor)
            if save_path:
                img = Image.frombytes('RGB', img.size, img.bgra, 'raw', 'BGRX')
                img.save(save_path + str(self.save_counter) + ".tiff")

            img = np.array(img)
            img = img[:, :, 0:3]
        return img

    def render_rays(self, img_arr, ray_list, angle):
        """creates a graphical representation of how the bot sees"""

        def make_back_and_white(img_arr):
            """sets pixel to black if the sum of rgb values are greater than colour threshold else white"""

            arr = (np.sum(img_arr, axis=2) < 200) * 255  # sum rgb and set to 0 or 255
            arr = np.expand_dims(arr, axis=2)  # place values in there own array [0,0] >> [[0],[0]]
            arr = np.repeat(arr, 3, axis=2)  # duplicate values to make r, g and b
            return arr

        def add_rays(self, img_arr, ray_list):
            """adds rays to image"""

            degrees_per_ray = self.degrees_per_ray
            ray_starting_distance = self.ray_starting_distance  # px
            ray_length = self.ray_length  # px
            ray_point_space = self.ray_point_space
            center_point = self.center_point
            rays_per_quadrant = int(90 / degrees_per_ray)

            for quadrant in range(4):
                for ray in range(rays_per_quadrant):
                    for ray_point in range(int((ray_length - ray_starting_distance) / ray_point_space)):

                        if quadrant in [1, 3]:
                            ray_actual = rays_per_quadrant - ray
                        else:
                            ray_actual = ray

                        point_distance = ray_point * ray_point_space + ray_starting_distance

                        x = int(np.sin(ray_actual * degrees_per_ray * np.pi / 180) * point_distance)
                        y = int(np.cos(ray_actual * degrees_per_ray * np.pi / 180) * point_distance)

                        if quadrant in [2, 3]:
                            x = -x
                        if quadrant in [1, 2]:
                            y = -y

                        y += center_point["y"]
                        x += center_point["x"]

                        if ray_list[quadrant * rays_per_quadrant + ray][ray_point] == 0:
                            colour = [0, 0, 0]
                        elif ray_list[quadrant * rays_per_quadrant + ray][ray_point] == 1:
                            colour = [0, 0, 255]
                        elif ray_list[quadrant * rays_per_quadrant + ray][ray_point] == 2:
                            colour = [255, 0, 0]
                        elif ray_list[quadrant * rays_per_quadrant + ray][ray_point] == 3:
                            colour = [0, 255, 0]
                        elif ray_list[quadrant * rays_per_quadrant + ray][ray_point] == 6:
                            colour = [0, 255, 255]

                        for i in range(3):
                            for j in range(3):
                                img_arr[y - 1 + i][x - 1 + j] = colour
            return img_arr

        def add_travel_angle_display(self, img_arr, angle):

            y = int(np.cos(angle * np.pi / 180) * 200) + self.center_point["y"]
            x = int(np.sin(angle * np.pi / 180) * 200) + self.center_point["x"]

            for i in range(20):
                for j in range(20):
                    img_arr[y - 10 + i][x - 10 + j] = [255, 0, 0]
            return img_arr

        img_arr = make_back_and_white(img_arr)
        img_arr = add_rays(self, img_arr, ray_list)
        img_arr = add_travel_angle_display(self, img_arr, angle)
        # img_arr = add_run_bool_display(img_arr, run_bool)

        return img_arr

    def get_optimum_angle(self, ray_list, current_direction):
        """calculates best direction to take"""

        def angle_differance(angle_1, angle_2):
            """returns the difference of angle_1 and angle_2 while considering crossing 0-360 degree range"""

            direction = 0
            if abs(angle_1 - angle_2) > 180:  # if difference crosses 0 list_positions_of_ones_in_cluster
                if angle_1 > angle_2:
                    direction = +1
                    distance = 360 - angle_1 + angle_2
                else:
                    direction = -1
                    distance = 360 - angle_2 + angle_1
                return distance * direction

            else:
                if angle_1 < angle_2:
                    direction = +1
                    distance = angle_2 - angle_1
                else:
                    direction = -1
                    distance = angle_1 - angle_2

            return distance * direction

        food_angle = 0
        preditor_angle = 0
        run = False

        for i, circle in enumerate(np.array(ray_list).T):
            for j, point in enumerate(circle):

                if food_angle == 0 and point == 2:
                    food_angle = ((j / len(ray_list)) * 360)

                if preditor_angle == 0 and point == 3:
                    preditor_angle = ((j / len(ray_list)) * 360)
                    break

            if preditor_angle != 0:
                break

        if preditor_angle != 0:
            angle = preditor_angle + 180
        elif food_angle != 0:
            angle = food_angle
        else:
            angle = current_direction

        if angle > 360:
            angle += -360

        angle_diff = angle_differance(current_direction, angle)

        if abs(angle_diff) > self.maximum_turn_per_frame:
            if angle_diff > 0:
                angle = current_direction + self.maximum_turn_per_frame
            else:
                angle = current_direction - self.maximum_turn_per_frame

        if angle > 360:
            angle += -360
        elif angle < 0:
            angle += 360

        return angle

    def mouse_pos(self, angle):
        """converts angle to mouse coordinates and moves mouse to that point"""

        x = int(np.sin(angle * np.pi / 180) * 100) + self.center_point["x"]
        y = int(np.cos(angle * np.pi / 180) * 100) + self.center_point["y"]

        self.mouse.move(x, y)

    def render(self):
        """gets image ray_list and angle at save and renders"""
        i = 0
        while True:
            if not os.path.exists("capture/frame" + str(i) + ".tiff"):  # exits when finished
                print("breaking")
                break

            img_arr = self.open_img("capture/frame" + str(i) + ".tiff")
            data_arr = pickle.load(open("capture/ray_angle_run" + str(i), 'rb'))
            new_img_arr = self.render_rays(img_arr, data_arr[0], data_arr[1])
            img = Image.fromarray(new_img_arr.astype('uint8'))
            f = open("capture/text" + str(i), 'a')
            if self.debug:
                for z in data_arr[0]:
                    f.write(str(z) + "\n")
                f.write(str(data_arr[1]) + "\n")
                f.close()
            img.save("capture/render" + str(i) + ".jpeg")
            i += 1
            print(i)

    def play(self):

        current_angle = 0

        capture_frame_bool = False

        capture_timer = 0
        frame_timer = 0

        while True:

            if self.frame_capture_rate != None and time.time() - self.frame_capture_rate > capture_timer:
                capture_timer = time.time()
                capture_frame_bool = True
            else:
                capture_frame_bool = False

            img = self.capture_frame("capture/frame" if capture_frame_bool else None)

            ray_list = self.get_rays(img)
            ray_list = self.food_or_preditor(ray_list)
            ray_list = self.remove_food_on_sides(ray_list, current_angle)
            angle = self.get_optimum_angle(ray_list, current_angle)

            self.mouse_pos(angle)

            if capture_frame_bool:
                save = [ray_list, angle]
                pickle.dump(save, open("capture/ray_angle_run" + str(self.save_counter), 'wb'))
                print("save " + str(self.save_counter))
                self.save_counter += 1

            frame_timer = time.time()

            current_angle = angle


snake = slither_bot()
# snake.render()
snake.play()
