import os
import imageio
import torch
import torch.nn.functional as F
from utils.environment_util import EnvState
import numpy as np
from utils.projection_utils import put_keypoints_on_image, get_set_of_vertices_projection


def save_image_list_to_gif(image_list, gif_name, gif_dir):
    gif_adr = os.path.join(gif_dir, gif_name)

    seq_len, cols, w, h, c = image_list.shape

    pallet = torch.zeros((seq_len, w, h * cols, c))

    for col_ind in range(cols):
        pallet[:, :, col_ind * h: (col_ind + 1) * h, :] = image_list[:, col_ind]

    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    imageio.mimsave(gif_adr, (pallet * 255.).type(torch.uint8), format='GIF', duration=1 / 5)
    print('Saved result in ', gif_adr)


def get_image_list_for_forces(full_output_position, full_output_rotation, force_applied, gt_contact_point, rgb_image, object_name, environment):
    rgb_image = rgb_image[:-1]

    all_images = []
    for seq_ind in range(force_applied.shape[0]):
        env_state = EnvState(position=full_output_position[seq_ind], rotation=full_output_rotation[seq_ind], object_name=object_name)
        output_image = environment.get_rgb_for_force(object_state=env_state, contact_point=gt_contact_point, force=force_applied[seq_ind])
        # combine images
        # combined_images = np.stack([output_image], axis=0)
        all_images.append(output_image)

    # stack images on top, base on numpy or pytorch. it has to be of shape # batch, time, channels, height, width,
    image_list = np.stack(all_images, axis=1)

    # convert to tensor and make it channel first, we have to convert it to [0,1] because tensorboardX does not work with 0-255
    image_list = torch.Tensor(image_list).permute(0, 1, 4, 2, 3).float() / 255.

    image_shapes = (image_list.shape[-2], image_list.shape[-1])

    # adding rgb images
    rgb_image = (F.interpolate(normalize(rgb_image), size=image_shapes))
    rgb_image = rgb_image.unsqueeze(0)
    image_list = torch.cat([rgb_image, image_list], dim=0)
    image_list = image_list.permute(1, 0, 3, 4, 2)
    image_list = add_border_to_images(image_list)
    return image_list


def get_image_list_for_viz_cp(output_cp, target_cp, full_target_position, full_target_rotation, object_name, input_rgb, environment):
    all_images = []
    for seq_ind in range(output_cp.shape[0]):
        env_state = EnvState(position=full_target_position[seq_ind], rotation=full_target_rotation[seq_ind], object_name=object_name)
        target_image = environment.get_rgb_for_position_rotation(object_state=env_state, contact_point=target_cp, contact_points_size=1)
        output_image = environment.get_rgb_for_position_rotation(object_state=env_state, contact_point=output_cp[seq_ind], contact_points_size=1)
        # combine images
        combined_images = np.stack([target_image, output_image], axis=0)
        all_images.append(combined_images)

    # stack images on top, base on numpy or pytorch. it has to be of shape # batch, time, channels, height, width,
    image_list = np.stack(all_images, axis=1)

    # convert to tensor and make it channel first, we have to convert it to [0,1] because tensorboardX does not work with 0-255
    image_list = torch.Tensor(image_list).permute(0, 1, 4, 2, 3).float() / 255.

    image_shapes = (image_list.shape[-2], image_list.shape[-1])

    # adding rgb images
    rgb_image = (F.interpolate(normalize(input_rgb), size=image_shapes))
    rgb_image = rgb_image.unsqueeze(0)
    image_list = torch.cat([rgb_image, image_list], dim=0)
    image_list = image_list.permute(1, 0, 3, 4, 2)
    image_list = add_border_to_images(image_list)
    return image_list


def draw_mesh_overlay(input, output, target, environment):
    output_rotation = output['rotation'][0]
    output_position = output['position'][0]
    target_rotation = target['rotation'][0]
    target_position = target['position'][0]
    object_name = input['object_name'][0]

    # adding the first frame to all of them

    initial_position = input['initial_position']
    initial_rotation = input['initial_rotation']

    full_output_position = torch.cat([initial_position, output_position], dim=0)
    full_output_rotation = torch.cat([initial_rotation, output_rotation], dim=0)
    full_target_position = torch.cat([initial_position, target_position], dim=0)
    full_target_rotation = torch.cat([initial_rotation, target_rotation], dim=0)

    image_shapes = (environment.qualitative_size, environment.qualitative_size)

    # adding rgb images
    rgb_image = (F.interpolate(normalize(input['rgb'][0]), size=image_shapes))
    rgb_image = rgb_image.unsqueeze(0)

    non_batched_rgb = rgb_image.squeeze(0)

    all_images = []
    for seq_ind in range(full_output_rotation.shape[0]):
        def get_mesh_overlay_projection(object_name, position, rotation, rgb, environment):
            rgb = rgb + 0.  # copy image
            env = environment.list_of_envs[object_name]
            set_of_points = get_set_of_vertices_projection(env, position, rotation)
            image = put_keypoints_on_image(rgb, set_of_points, SIZE_OF_DOT=4, coloring=False)
            return image

        output_mesh_overlay = get_mesh_overlay_projection(object_name, full_output_position[seq_ind], full_output_rotation[seq_ind], non_batched_rgb[seq_ind], environment)
        target_mesh_overlay = get_mesh_overlay_projection(object_name, full_target_position[seq_ind], full_target_rotation[seq_ind], non_batched_rgb[seq_ind], environment)

        # combine images
        combined_images = torch.stack([target_mesh_overlay, output_mesh_overlay], dim=0)

        all_images.append(combined_images)

    # stack images on top, base on numpy or pytorch. it has to be of shape # batch, time, channels, height, width,
    image_list = torch.stack(all_images, dim=1)

    image_list = torch.cat([rgb_image, image_list], dim=0)
    image_list = image_list.permute(1, 0, 3, 4, 2)
    image_list = add_border_to_images(image_list)

    return image_list


def get_image_list_for_keypoints(full_target_keypoints, full_output_keypoints, full_output_position, full_output_rotation, gt_contact_point, rgb_image, object_name, environment):
    # adding rgb images
    rgb_image = (F.interpolate(normalize(rgb_image), size=(environment.qualitative_size, environment.qualitative_size))).cpu().detach()
    rgb_image = rgb_image.unsqueeze(0)

    rgb_image_channel_last = rgb_image.squeeze(0).permute(0, 2, 3, 1)

    all_images = []
    for seq_ind in range(full_output_position.shape[0]):
        output_env_state = EnvState(position=full_output_position[seq_ind], rotation=full_output_rotation[seq_ind], object_name=object_name)
        output_image = environment.get_rgb_for_position_rotation(object_state=output_env_state, contact_point=gt_contact_point)  # add force viz, forces_poc=force_visualization[seq_ind])
        output_image = torch.Tensor(output_image).float() / 255.
        image_for_target_keypoints = put_keypoints_on_image(rgb_image_channel_last[seq_ind], full_target_keypoints[seq_ind])
        image_for_output_keypoints = put_keypoints_on_image(rgb_image_channel_last[seq_ind], full_output_keypoints[seq_ind])

        combined_images = torch.stack([image_for_target_keypoints, image_for_output_keypoints, output_image], dim=0)

        all_images.append(combined_images)

    # stack images on top, base on numpy or pytorch. it has to be of shape # batch, time, channels, height, width,
    image_list = torch.stack(all_images, dim=1)

    # convert to tensor and make it channel first, we have to convert it to [0,1] because tensorboardX does not work with 0-255
    image_list = image_list.permute(0, 1, 4, 2, 3).float()

    image_list = torch.cat([rgb_image, image_list], dim=0)
    image_list = image_list.permute(1, 0, 3, 4, 2)
    image_list = add_border_to_images(image_list)
    return image_list


def get_image_list_for_object_traj(output_rotation, output_position, target_rotation, target_position, gt_contact_point, initial_rotation, initial_position, object_name, environment, input_rgb, output_contact_point=None):
    env_state = EnvState(position=initial_position, rotation=initial_rotation, object_name=object_name)

    initial_image = environment.get_rgb_for_position_rotation(object_state=env_state)

    initial_double_image = np.stack([initial_image, initial_image], axis=0)  # because we need it both for target and output

    all_images = [initial_double_image]
    for seq_ind in range(output_rotation.shape[0]):
        output_env_state = EnvState(position=output_position[seq_ind], rotation=output_rotation[seq_ind], object_name=object_name)
        output_image = environment.get_rgb_for_position_rotation(object_state=output_env_state, contact_point=output_contact_point)  # add force viz, forces_poc=force_visualization[seq_ind])
        target_env_state = EnvState(position=target_position[seq_ind], rotation=target_rotation[seq_ind], object_name=object_name)
        target_image = environment.get_rgb_for_position_rotation(object_state=target_env_state, contact_point=gt_contact_point)

        # combine images
        combined_images = np.stack([target_image, output_image], axis=0)

        all_images.append(combined_images)

    # stack images on top, base on numpy or pytorch. it has to be of shape # batch, time, channels, height, width,
    image_list = np.stack(all_images, axis=1)

    # convert to tensor and make it channel first, we have to convert it to [0,1] because tensorboardX does not work with 0-255
    image_list = torch.Tensor(image_list).permute(0, 1, 4, 2, 3).float() / 255.

    image_shapes = (image_list.shape[-2], image_list.shape[-1])

    # adding rgb images
    rgb_image = (F.interpolate(normalize(input_rgb), size=image_shapes))
    rgb_image = rgb_image.unsqueeze(0)
    image_list = torch.cat([rgb_image, image_list], dim=0)
    image_list = image_list.permute(1, 0, 3, 4, 2)
    image_list = add_border_to_images(image_list)

    return image_list


def channel_first(img):
    if type(img) == torch.Tensor:
        return img.transpose(-1, -2).transpose(-2, -3)
    if type(img) == np.ndarray:
        return img.swapaxes(-1, -2).swapaxes(-2, -3)


def channel_last(img):
    if type(img) == torch.Tensor:
        return img.transpose(-3, -2).transpose(-2, -1)
    if type(img) == np.ndarray:
        return img.swapaxes(-3, -2).swapaxes(-2, -1)


def normalize(img):
    img = img.cpu()
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])
    img = channel_last(img)
    img = (img * std + mean)
    img = torch.clamp(img, 0, 1)
    return channel_first(img)


def add_border_to_images(img_list, pixel=5):
    seq_len, num_imgs, w, h, c = img_list.shape
    assert c == 3
    assert torch.is_tensor(img_list)
    img = img_list + 0
    img[:, :, pixel, :] = 0
    img[:, :, -pixel:, :] = 0
    img[:, :, :, :pixel] = 0
    img[:, :, :, -pixel:] = 0
    return img
