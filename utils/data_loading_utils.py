import json
from .constants import ALL_OBJECTS, OBJECT_TO_SCALE
import torch
import datetime


def process_projection(dict_of_keypoints):
    result = []
    for kp_ind in range(10):
        kp_ind = str(kp_ind)
        if kp_ind in dict_of_keypoints and dict_of_keypoints[kp_ind]['label'] <= 2:
            result.append([dict_of_keypoints[kp_ind]['x'], dict_of_keypoints[kp_ind]['y']])
        else:
            result.append([1e-10, 1e-10])
    return result


def scale_position(position, object_name):
    scale = OBJECT_TO_SCALE[object_name]
    if not torch.is_tensor(position):
        position = torch.Tensor(position)
    return position * scale


def _load_transformation_files(json_file):
    with open(json_file) as f:
        json_dict = json.load(f)
    return json_dict


def obtain_object_name(img_adr):
    obj_name = [obj for obj in ALL_OBJECTS if obj in img_adr]
    assert len(obj_name) == 1
    return obj_name[0]


def divide_objects(dict_file):
    result_dict = {}
    for time, dict in dict_file.items():
        obj_name = obtain_object_name(dict['image_url'])
        result_dict.setdefault(obj_name, {})
        result_dict[obj_name][parse_time_str(time)] = dict
    return result_dict


def parse_time_str(timestr):
    time = datetime.datetime.strptime(timestr, '%Y-%m-%d %H:%M:%S.%f')
    return time


def obtain_time_from_img_name(url):
    image_name = url.split('/')[-1].replace('.jpeg', '')
    timestamp = '_'.join(image_name.split('_')[2:])
    time = datetime.datetime.strptime(timestamp, '%Y_%m_%d_%H_%M_%S_%f')
    return time


def get_timestamp_to_clip_index(clip_dict):
    result = {}
    for obj in clip_dict.keys():
        result[obj] = {}
        for clip_ind, dict in clip_dict[obj]['clips'].items():
            img_list = dict['list_of_images']
            time_stamps = [obtain_time_from_img_name(img) for img in img_list]
            dict = {time: clip_ind for time in time_stamps}
            result[obj].update(dict)
    return result


def get_time_from_str(time):
    return datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S.%f')
