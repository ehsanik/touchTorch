# Code based on examples from https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/
# See gym env base file for specifications on API: https://github.com/openai/gym/blob/master/gym/core.py

import numpy as np

import gym
from gym.utils import seeding
import pybullet

from environments import bullet_client

# Note: Environment below is only compatible with gym >= 0.9.6
from pkg_resources import parse_version

assert (parse_version(gym.__version__) >= parse_version('0.9.6'))


## Basic environment with rendering functionality and shared memory
class BaseBulletEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, render=False):
        '''
        Basic environment initialization.
        
        Child classes need to define:
            action_space
            observation_space
            step()
            reset()
            close()
            seed()
        
        '''
        self._is_render = render
        if self._is_render:
            self._p = bullet_client.BulletClient(
                connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()

        self._cam_dist = 3
        self._cam_yaw = 0
        self._cam_pitch = -30
        self._render_width = 320
        self._render_height = 240

    def seed(self, seed=None):
        self._rng, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='rgb_array', renderer=pybullet.ER_TINY_RENDERER):
        if mode != 'rgb_array':
            return np.array([])

        base_pos = [0, 0, 0]
        if hasattr(self, 'robot') and hasattr(self.robot, 'body_xyz'):
            base_pos = self.robot.body_xyz

        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos, distance=self._cam_dist,
            yaw=self._cam_yaw, pitch=self._cam_pitch, roll=0, upAxisIndex=2)

        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60, aspect=float(self._render_width) / self._render_height,
            nearVal=0.1, farVal=100.0)

        (_, _, px, _, _) = self._p.getCameraImage(
            width=self._render_width, height=self._render_height, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=renderer)

        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        # : not sure if this is correct!
        self._p.disconnect()
