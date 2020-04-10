from environments.base_env import BaseBulletEnv

import pybullet_data
from utils.transformations import quaternion_normal2bullet, quaternion_bullet2normal

import os
import time
import numpy as np
import torch
from utils.obj_util import obtain_all_vertices_from_obj
from utils.quaternion_util import quaternion_to_rotation_matrix
from utils.environment_util import EnvState
from utils.constants import OBJECT_TO_SCALE, CONTACT_POINT_MASK_VALUE, GRAVITY_VALUE


class PhysicsEnv(BaseBulletEnv):

    def __init__(self, args, object_path, object_name):
        super(PhysicsEnv, self).__init__(render=args.render)
        self._pb_urdf_root = pybullet_data.getDataPath()
        self.object_name = object_name
        self.render = args.render
        self.object_path = object_path
        self.continue_loading = False
        self.gravity = args.gravity
        self.debug = args.debug
        self.pybullet_data_path = pybullet_data.getDataPath()
        self.time_step_length = 1 / 240.
        self.number_of_cp = args.number_of_cp
        self.gpu_ids = args.gpu_ids
        self.fps = args.fps
        self.number_of_steps_per_image = int(1 / self.fps / self.time_step_length)
        self.cameraEyePosition = np.array([0, 0, 0])
        self.camera_gaze_direction = np.array([0, 0, 1])
        self.force_multiplier = args.force_multiplier
        self.force_h = args.force_h
        self.state_h = args.state_h
        self.object_path = object_path
        self.scale = OBJECT_TO_SCALE[object_name]
        if self.render:
            assert args.workers == 0
        self.sleep_time = 1
        # assert self.gravity
        self.qualitative_size = args.qualitative_size

    def reset(self):

        self.terminated = 0
        self._p.resetSimulation()
        self._p.setRealTimeSimulation(0)
        self._p.setPhysicsEngineParameter(numSolverIterations=150)  # , enableConeFriction=0)
        self._p.setTimeStep(self.time_step_length)
        if self.gravity:
            self._p.setGravity(GRAVITY_VALUE[0], GRAVITY_VALUE[1], GRAVITY_VALUE[2])

        self.object_of_interest_id, self.vertex_points, self.faces, self.middle_points, self.vertex_to_faces_dict, self.center_of_mass = self.initiate_object(
            self.object_path)
        self.center_of_mass = torch.Tensor(self.center_of_mass)
        self.vertex_points = self.vertex_points - self.center_of_mass
        if self.gpu_ids != -1:
            self.center_of_mass = self.center_of_mass.cuda()
            self.vertex_points = self.vertex_points.cuda()
            # self.faces, self.middle_points are not on cuda

        self.on_plane_points = [self.vertex_to_faces_dict[i][0].tolist() for i in range(len(self.vertex_to_faces_dict))]
        self.on_plane_points = torch.Tensor(self.on_plane_points).long()
        if self.gpu_ids != -1:
            self.on_plane_points = self.on_plane_points.cuda()
        self.all_surface_normals = self.calculate_surface_normals(self.vertex_points, self.vertex_to_faces_dict)

    def calculate_surface_normals(self, vertices, vertex_to_face_dict):
        all_normals = []
        for vert_ind in range(len(vertices)):
            face = vertex_to_face_dict[vert_ind][0].tolist()
            vertices_on_this_face = vertices[face]
            surface_normal = torch.cross((vertices_on_this_face[1] - vertices_on_this_face[0]) * 10, (vertices_on_this_face[2] - vertices_on_this_face[0]) * 10)  # To avoid numerical errors because these are all very close
            surface_normal = surface_normal / surface_normal.norm()
            all_normals.append(surface_normal)
        normal_end_vec = torch.stack(all_normals, dim=0)
        normal_begin_vec = torch.zeros(normal_end_vec.shape, device=normal_end_vec.device)
        return torch.stack([normal_begin_vec, normal_end_vec], dim=1)  # Just adding a zero to it, this will be useful for calculating the new normals

    def check_force_hit(self, contact_point, surface_normal, force_value):
        d = -(contact_point * surface_normal).sum()
        force_secondary_point = contact_point + force_value
        side_of_force = (force_secondary_point * surface_normal).sum() + d
        if side_of_force < 0:
            # force is applied directly
            return 1, force_value
        elif side_of_force > 0:
            # force is applied tangentically
            a_cross_b = torch.cross(force_value, surface_normal)
            projected_force = torch.cross(surface_normal, a_cross_b / surface_normal.norm()) / surface_normal.norm()
            return 0.5, projected_force
        else:
            return 0, None

    def initiate_object(self, object_path):
        unique_id = self._p.loadURDF(object_path, basePosition=[0.5, 0.5, 0.5], baseOrientation=[1, 0, 0, 0], globalScaling=self.scale)
        center_of_mass = self._p.getDynamicsInfo(bodyUniqueId=unique_id, linkIndex=-1)[3]
        self._p.changeDynamics(bodyUniqueId=unique_id, linkIndex=-1, mass=0.02, spinningFriction=0.001)  # This is just to make sure is the same as the block
        with open(object_path, 'r') as f:
            lines = [l for l in f]
            simple_convex_line = [l for l in lines if '_convex' in l]
            if len(simple_convex_line) == 0:
                simple_convex_line = [l for l in lines if 'bigconvex' in l]
            assert len(simple_convex_line) == 1
            simple_convex_file_name = [l for l in simple_convex_line[0].split('\"') if '.obj' in l][0]
            simple_convex_file_name = os.path.join(('/'.join(object_path.split('/')[:-1])), simple_convex_file_name)
        vertices, faces, middles_pts = obtain_all_vertices_from_obj(simple_convex_file_name)

        vertex_to_faces_dict = {}
        for face in faces:
            for f in face:
                vertex_to_faces_dict.setdefault(f, [])
                vertex_to_faces_dict[f].append(face)
        return unique_id, torch.Tensor(vertices) * self.scale, torch.Tensor(faces) * self.scale, torch.Tensor(middles_pts) * self.scale, vertex_to_faces_dict, center_of_mass

    def reset_base_with_normal_quaternion(self, bodyUniqueId, posObj, ornObj):
        assert torch.abs(1 - ((ornObj).norm(2))) < 1e-5
        ornObj = quaternion_normal2bullet(ornObj)
        self._p.resetBasePositionAndOrientation(bodyUniqueId=bodyUniqueId, posObj=posObj, ornObj=ornObj)

    def update_object_transformations(self, object_state, object_num, hand_pose=None):

        rotation = object_state.rotation
        position = object_state.position
        velocity = object_state.velocity
        omega = object_state.omega
        self.reset_base_with_normal_quaternion(bodyUniqueId=object_num, posObj=position, ornObj=rotation)
        self._p.resetBaseVelocity(objectUniqueId=object_num, linearVelocity=velocity, angularVelocity=omega)

    def get_rgb(self):
        output_w, output_h = self.qualitative_size, self.qualitative_size

        # To get the matrices:
        #
        #     width, height, viewMat, projMat, cameraUp, camForward, horizon, vertical, yaw, pitch, dist, camTarget = self._p.getDebugVisualizerCamera()
        #
        #     self._p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=yaw, cameraPitch=pitch, cameraTargetPosition=camTarget)
        #
        #     viewMat = self._p.computeViewMatrix(cameraEyePosition=[0, 0, 0], cameraTargetPosition=[0,0,1], cameraUpVector=[0, -1, 0])
        #
        #     w, h, rgb, depth, segmmask = self._p.getCameraImage(output_w, output_h, viewMatrix=viewMat, projectionMatrix=projMat)

        viewMat = (1.0, 0.0, -0.0, 0.0, 0.0, -1.0, -0.0, 0.0, -0.0, 0.0, -1.0, 0.0, -0.0, -0.0, 0.0, 1.0)
        projMat = (0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0)
        w, h, rgb, depth, segmmask = self._p.getCameraImage(output_w, output_h, viewMatrix=viewMat, projectionMatrix=projMat)

        return rgb[:, :, :3]

    def find_vertices_on_mesh_using_cp(self, contact_points):
        MAXIMUM_CP_FINGER_DISTANCE = 1
        contact_points = contact_points - self.center_of_mass
        all_distances = self.vertex_points.unsqueeze(0).repeat((5, 1, 1)) - contact_points.unsqueeze(1)
        norm_distances = all_distances.norm(dim=-1)
        cp_on_mesh_ind = torch.argmin(norm_distances, dim=-1)
        for cp_ind in range(self.number_of_cp):
            dist = norm_distances[cp_ind, cp_on_mesh_ind[cp_ind]]
            if dist > MAXIMUM_CP_FINGER_DISTANCE:
                cp_on_mesh_ind[cp_ind] = -1
        return cp_on_mesh_ind.detach()

    def step(self):
        self._p.stepSimulation()
        if self.render:
            time.sleep(1 / 200)

    def convert_set_of_vertices(self, vertices, position, rotation_mat):
        vertices = vertices.transpose(0, 1)
        rotated_vertices = torch.mm(rotation_mat, vertices).transpose(0, 1)
        return rotated_vertices + position.view(1, 3)

    def get_rotation_mat(self, state):
        return quaternion_to_rotation_matrix(state.rotation.view(1, 4)).squeeze(0)

    def get_current_state(self, object_num=None):
        if object_num is None:
            object_num = self.object_of_interest_id
        objectPos, objectOr = self._p.getBasePositionAndOrientation(object_num)
        objectOr = quaternion_bullet2normal(objectOr)
        velocity, omega = self._p.getBaseVelocity(bodyUniqueId=object_num)
        return EnvState(object_name=self.object_name, position=objectPos, rotation=objectOr, velocity=velocity, omega=omega)

    def init_location_and_apply_force(self, forces, initial_state, object_num, list_of_contact_points):

        if list_of_contact_points is None:
            gap = int(len(self.vertex_points) / self.number_of_cp)
            list_of_contact_points = [i * gap for i in range(self.number_of_cp)]
        else:
            list_of_contact_points = self.find_vertices_on_mesh_using_cp(list_of_contact_points)

        if object_num is None:
            object_num = self.object_of_interest_id

        self.update_object_transformations(object_state=initial_state, object_num=object_num)

        overall_success = []

        list_of_force_success = []
        list_of_force_location = []
        list_of_applied_forces = []

        contact_points = self.vertex_points[list_of_contact_points]

        latest_state = self.get_current_state()
        if self.gpu_ids != -1:
            latest_state.cuda_()

        latest_rotation_mat = self.get_rotation_mat(latest_state)
        latest_translation = latest_state.position
        converted_cp = self.convert_set_of_vertices(contact_points, latest_translation, latest_rotation_mat)

        for cp_ind in range(self.number_of_cp):
            if list_of_contact_points[cp_ind] == -1:
                force_success = False
                contact_point = torch.Tensor([CONTACT_POINT_MASK_VALUE, CONTACT_POINT_MASK_VALUE, CONTACT_POINT_MASK_VALUE])
                force_value = torch.Tensor([-1e10, -1e10, -1e10])
                if self.gpu_ids != -1:
                    contact_point = contact_point.cuda()
                    force_value = force_value.cuda()
                list_of_force_success.append(force_success)
                list_of_force_location.append(contact_point)
                list_of_applied_forces.append(force_value)
                continue  # we should not get a penalty for predicting force on it
            force_to_apply = forces[cp_ind].force
            contact_point = converted_cp[cp_ind]

            unoriented_surface_normal = self.all_surface_normals[list_of_contact_points[cp_ind]]  # Remember this is not a copy this is the real one, DO NOT CHANGE THIS
            surface_normal_end_begin = self.convert_set_of_vertices(unoriented_surface_normal, latest_translation, latest_rotation_mat)  # checked this visually
            surface_normal = surface_normal_end_begin[1] - surface_normal_end_begin[0]
            d = -(contact_point * surface_normal).sum()
            center_situation = (latest_state.position * surface_normal).sum() + d
            if center_situation > 0:
                surface_normal *= -1

            force_success, force_value = self.apply_force_to_obj(force_to_apply=force_to_apply, contact_point=contact_point, surface_normal=surface_normal, object_num=object_num)

            list_of_force_success.append(force_success)
            list_of_force_location.append(contact_point)
            list_of_applied_forces.append(force_value)

        overall_success.append(list_of_force_success)

        for step_number in range(self.number_of_steps_per_image):
            self.step()  # Do not remove this

        current_state = self.get_current_state()

        assert len(forces) == self.number_of_cp

        return current_state, list_of_force_success, list_of_force_location

    def apply_force_to_obj(self, force_to_apply, contact_point, surface_normal, object_num=None):
        poc = contact_point
        if object_num is None:
            object_num = self.object_of_interest_id

        if self.render and force_to_apply.abs().sum() == 0:
            force_success = 0
            force_applied = force_to_apply
        else:
            hit_, force_applied = self.check_force_hit(contact_point=poc, force_value=force_to_apply, surface_normal=surface_normal)
            force_success = hit_

            if hit_ > 0:
                self._p.applyExternalForce(objectUniqueId=object_num, linkIndex=-1, posObj=contact_point, forceObj=force_to_apply * self.force_multiplier, flags=self._p.WORLD_FRAME)

        return force_success, force_applied
