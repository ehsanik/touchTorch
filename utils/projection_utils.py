import json
from utils.constants import ALL_OBJECT_KEYPOINT_NAME, ALL_OBJECTS, OBJECT_TO_SCALE, OBJECT_NAME_TO_CENTER_OF_MASS, DISTORTION, CAMERA_INTRINS_UNNORM, DEFAULT_IMAGE_SIZE
import pdb
import torch
from utils.quaternion_util import quaternion_to_rotation_matrix
import os

def _get_object_keypoints(object_name, root_path):
    path = os.path.join(root_path, ALL_OBJECT_KEYPOINT_NAME[object_name])
    with open(path) as file:
        keypoint_dict = json.load(file)
    keypoint_list = [keypoint_dict[str(kp_ind)] for kp_ind in range(10)]
    return torch.Tensor(keypoint_list)


def get_all_objects_keypoint_tensors(dataset_path):
    return {obj_name: _get_object_keypoints(obj_name, dataset_path) for obj_name in ALL_OBJECTS}


def project_points(points, rotation_mat, translation_mat, camera_intr, distortion):
    k1, k2, p1, p2, k3 = distortion
    xyz = torch.mm(rotation_mat, points.transpose(0, 1)).transpose(0, 1) + translation_mat
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    xp = x / z
    yp = y / z
    r2 = xp ** 2 + yp ** 2
    r4 = r2 ** 2
    r6 = r2 ** 3
    radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
    xpp = xp * radial + 2. * p1 * xp * yp + p2 * (r2 + 2 * xp ** 2)
    ypp = yp * radial + p1 * (r2 + 2 * yp ** 2) + 2. * p2 * xp * yp
    u = camera_intr[0, 0] * xpp + camera_intr[0, 2]
    v = camera_intr[1, 1] * ypp + camera_intr[1, 2]
    return torch.stack([u, v], dim=-1)

def reverse_translation_transformation(translation_mat, object_name):
    center_of_mass = OBJECT_NAME_TO_CENTER_OF_MASS[object_name]
    if center_of_mass.device != translation_mat.device:
        center_of_mass = center_of_mass.to(translation_mat.device)
    scale = OBJECT_TO_SCALE[object_name]
    translation_mat = translation_mat / scale + center_of_mass.detach()
    return translation_mat

def get_keypoint_projection(object_name, resulting_positions, resulting_rotations, keypoints):
    batch_size, seq_len, size = resulting_positions.shape
    assert batch_size == 1
    resulting_positions = resulting_positions.squeeze(0)
    resulting_rotations = resulting_rotations.squeeze(0)
    # keypoints = ALL_OBJECTS_KEYPOINTS_TENSORS[object_name]
    if keypoints.device != resulting_positions.device:
        keypoints = keypoints.to(resulting_positions.device)
    resulting_positions = reverse_translation_transformation(resulting_positions, object_name)
    rotation_mats = quaternion_to_rotation_matrix(resulting_rotations)
    all_projections = []
    for seq_ind in range(seq_len):
        rotation_mat = rotation_mats[seq_ind]
        translation_mat = resulting_positions[seq_ind]
        result = project_points(keypoints, rotation_mat, translation_mat, CAMERA_INTRINS_UNNORM, DISTORTION)
        all_projections.append(result)
    return torch.stack(all_projections, dim=0)

def put_a_dot_on_image(image, point, color, SIZE_OF_DOT):
    point = torch.round(point).long() + 0. #Copy point

    h, w , c = image.shape
    ch_first = False

    if h == 3 and c > 3:
        c, h, w = image.shape
        ch_first = True

    point[0] = max(point[0], 0 + SIZE_OF_DOT)
    point[0] = min(point[0], w - SIZE_OF_DOT)
    point[1] = max(point[1], 0 + SIZE_OF_DOT)
    point[1] = min(point[1], h - SIZE_OF_DOT)

    if ch_first:
        image[:, point[1] - SIZE_OF_DOT: point[1] + SIZE_OF_DOT, point[0] - SIZE_OF_DOT: point[0] + SIZE_OF_DOT] = torch.Tensor(color).unsqueeze(1).unsqueeze(2).float()
    else:
        image[point[1] - SIZE_OF_DOT: point[1] + SIZE_OF_DOT, point[0] - SIZE_OF_DOT: point[0] + SIZE_OF_DOT] = torch.Tensor(color).float()


def convert_to_color(kp_ind):
    all_colors = {
        0: (0,1.,0), #green
        1: (0,0,1.), #blue
        2: (1.,0.65,0), #orange
        3: (1.,0,0), #red
        4: (0.65, 0, 1.), #purple
        5: ((150./255.), (75./255.), 0), #brown
        6: (1, 105./255., 180./255.), #pink
        7: (128./255.,128./255.,128./255.), #gray
        8: (1,1.,0), #yellow
        9: (1,204./255.,153./255.), #light orange
    }
    return all_colors[kp_ind]
def put_keypoints_on_image(image, keypoints, SIZE_OF_DOT=5, coloring=True):
    image = image + 0. #Copy image
    w, h, c = image.shape
    if w == 3:
        c, w, h = image.shape
    image_shape = torch.Tensor([w, h]).float()
    image_size = DEFAULT_IMAGE_SIZE
    if image_size.device != keypoints.device:
        image_size = image_size.to(keypoints.device)
    converted_keypoints = keypoints / image_size * image_shape
    for kp_ind in range(converted_keypoints.shape[0]):
        if coloring:
            color = convert_to_color(kp_ind)
        else:
            color = convert_to_color(1)
        put_a_dot_on_image(image, converted_keypoints[kp_ind], color, SIZE_OF_DOT)
    return image



def reverse_center_of_mass_translation(translation_mat, environment):
    center_of_mass = environment.center_of_mass
    if center_of_mass.device != translation_mat.device:
        center_of_mass = center_of_mass.to(translation_mat.device)
    translation_mat = translation_mat - center_of_mass.detach()
    return translation_mat

def get_set_of_vertices_projection(environment, resulting_positions, resulting_rotations):
    set_of_points = environment.vertex_points
    if set_of_points.device != resulting_positions.device:
        set_of_points = set_of_points.to(resulting_positions.device)
    set_of_points = (set_of_points + environment.center_of_mass) / OBJECT_TO_SCALE[environment.object_name]
    # translation_mat = reverse_center_of_mass_translation(resulting_positions, environment)
    # translation_mat = resulting_positions
    translation_mat = reverse_translation_transformation(resulting_positions, environment.object_name)
    rotation_mat = quaternion_to_rotation_matrix(resulting_rotations.unsqueeze(0)).squeeze(0)
    result = project_points(set_of_points, rotation_mat, translation_mat, CAMERA_INTRINS_UNNORM, DISTORTION)

    return result