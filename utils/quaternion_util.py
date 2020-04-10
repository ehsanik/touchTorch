import torch
import torch.nn.functional as f
import pdb


def get_quaternion_distance(q1_array, q2_array):
    '''
    According to https://math.stackexchange.com/questions/90081/quaternion-distance
    and this explains why:
    https://fgiesen.wordpress.com/2013/01/07/small-note-on-quaternion-distance-metrics/
    This paper has L2 loss for quaternions and claims it works:
    http://mi.eng.cam.ac.uk/~cipolla/publications/inproceedings/2017-CVPR-posenet-geometric-loss.pdf
    the blog posts explains why
    " In particular, it gives 0 whenever the quaternions represent the same orientation, and it gives 1 whenever the two orientations are 180∘ apart."
    '''
    assert q1_array.shape == q2_array.shape
    assert q1_array.shape[-1] == 4
    final_shape = q1_array[..., 3].shape  # just getting rid of the last argument
    q1_array = q1_array.contiguous().view(-1, 4)
    q2_array = q2_array.contiguous().view(-1, 4)
    q1_array = normalize_quaternion(q1_array)
    q2_array = normalize_quaternion(q2_array)
    # how do you handle [0,0,0,0]? then the inner product is 0 and the diff is one and we are good to go
    inner_product = (q1_array * q2_array).sum(-1)
    diff = 1 - inner_product ** 2
    return diff.view(final_shape)


def quaternion_to_angle_axis(quaternion):
    # Borrowed from https://github.com/arraiyopensource/torchgeometry/blob/master/torchgeometry/core/conversions.py
    """Convert quaternion vector to angle axis of rotation.
    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h
    Args:
        quaternion (torch.Tensor): tensor with quaternions.
    Return:
        torch.Tensor: tensor with angle axis of rotation.
    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`
    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4 of shape w, x, y, z
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape Nx4 or 4. Got {}".format(
                quaternion.shape))
    # unpack input and compute conversion
    q1 = quaternion[..., 1]
    q2 = quaternion[..., 2]
    q3 = quaternion[..., 3]
    sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta = torch.sqrt(sin_squared_theta)
    cos_theta = quaternion[..., 0]
    two_theta = 2.0 * torch.where(
        cos_theta < 0.0, torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos = two_theta / sin_theta
    k_neg = 2.0 * torch.ones_like(sin_theta)
    k = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def subtract_quaternion(current_q, prev_q):
    assert len(current_q) == len(prev_q), "There needs to be equal number of IMUs"
    current_q = normalize_quaternion(current_q)
    prev_q = normalize_quaternion(prev_q)
    diff = multiply_quaternion(current_q, inverse_quaternion(prev_q))

    nan_detector = (diff != diff)
    nan_detector = nan_detector.sum(1) > 0
    if nan_detector.sum() > 0:
        pdb.set_trace()
        diff[nan_detector] = torch.Tensor([1, 0, 0, 0])
    zero_detector = (diff == torch.Tensor([0, 0, 0, 0]))
    zero_detector = (zero_detector.sum(1) == 4)
    if zero_detector.sum() > 0:
        pdb.set_trace()
        diff[zero_detector] = torch.Tensor([1, 0, 0, 0])

    diff = normalize_quaternion(diff)

    return diff


# borrowed from https://github.com/arraiyopensource/kornia/blob/d8c547152f93b56f7a8d6985be852ef5c9735650/kornia/geometry/conversions.py
# THis is differentiable
def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    r"""Converts a quaternion to a rotation matrix.
    The quaternion should be in (w, x, y, z) format.
    Args:
        quaternion (torch.Tensor): a tensor containing a quaternion to be
          converted. The tensor can be of shape :math:`(*, 4)`.
    Return:
        torch.Tensor: the rotation matrix of shape :math:`(*, 3, 3)`.
    Example:
        >>> quaternion = torch.tensor([0., 0., 1., 0.])
        >>> kornia.quaternion_to_rotation_matrix(quaternion)
        tensor([[[-1.,  0.,  0.],
                 [ 0., -1.,  0.],
                 [ 0.,  0.,  1.]]])
    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(
                quaternion.shape))
    # normalize the input quaternion
    quaternion_norm: torch.Tensor = normalize_quaternion(quaternion)

    # unpack the normalized quaternion components
    w, x, y, z = torch.chunk(quaternion_norm, chunks=4, dim=-1)

    # compute the actual conversion
    tx: torch.Tensor = 2.0 * x
    ty: torch.Tensor = 2.0 * y
    tz: torch.Tensor = 2.0 * z
    twx: torch.Tensor = tx * w
    twy: torch.Tensor = ty * w
    twz: torch.Tensor = tz * w
    txx: torch.Tensor = tx * x
    txy: torch.Tensor = ty * x
    txz: torch.Tensor = tz * x
    tyy: torch.Tensor = ty * y
    tyz: torch.Tensor = tz * y
    tzz: torch.Tensor = tz * z
    one: torch.Tensor = torch.tensor(1.)

    matrix: torch.Tensor = torch.stack([
        one - (tyy + tzz), txy - twz, txz + twy,
        txy + twz, one - (txx + tzz), tyz - twx,
        txz - twy, tyz + twx, one - (txx + tyy)
    ], dim=-1).view(-1, 3, 3)

    if len(quaternion.shape) == 1:
        matrix = torch.squeeze(matrix, dim=0)
    return matrix


def angle_axis_to_rotation_matrix(angle_axis):
    """Convert 3d vector of axis-angle rotation to 4x4 rotation matrix
    Args:
        angle_axis (Tensor): tensor of 3d vector of axis-angle rotations.
    Returns:
        Tensor: tensor of 4x4 rotation matrices.
    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 4, 4)`
    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = tgm.angle_axis_to_rotation_matrix(input)  # Nx4x4
    """

    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat(
            [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat(
            [k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)  # noqa

    # create output pose matrix
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.eye(4).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 4, 4).repeat(batch_size, 1, 1)
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = \
        mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix  # Nx4x4


def continue_quaternion(q_array, initial):
    assert (q_array.shape[2] == 4), "wrong format"
    all_q = [initial]
    for i in range(len(q_array)):
        next_q = multiply_quaternion(q_array[i], all_q[-1])
        all_q.append(next_q)
    result = torch.stack(all_q[1:])
    return normalize_quaternion(result)


def normalize_quaternion(q_array):
    assert q_array.shape[-1] == 4, "Wrong format"
    return f.normalize(q_array, dim=-1)


def inverse_quaternion(q_array):
    assert q_array.shape[1] == 4, "Wrong format"
    conjugate = conjugate_quaternion(q_array)
    norm2 = q_array.norm(dim=1, p=2)
    return conjugate / norm2.view(norm2.shape[0], 1)


def conjugate_quaternion(q_array):
    assert q_array.shape[1] == 4, "Wrong format"
    result = q_array * 1
    result[:, 1:] *= -1
    return result


def equality_quaternion(q1, q2):
    import quaternion
    THR = 1e-5
    return sum(abs(quaternion.as_float_array(q1) - q2.numpy())) < THR


def multiply_quaternion_optimized(q, r):
    """
    Borrowed from https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def multiply_quaternion(diff, original):
    assert diff.shape[1] == 4, "Wrong format"
    assert original.shape[1] == 4, "Wrong format"
    t = torch.Tensor(diff.shape)
    t[:, 0] = diff[:, 0] * original[:, 0] - diff[:, 1] * original[:, 1] - diff[:, 2] * original[:, 2] - diff[:, 3] * original[:, 3]
    t[:, 1] = original[:, 0] * diff[:, 1] + original[:, 1] * diff[:, 0] - original[:, 2] * diff[:, 3] + original[:, 3] * diff[:, 2]
    t[:, 2] = original[:, 0] * diff[:, 2] + original[:, 1] * diff[:, 3] + original[:, 2] * diff[:, 0] - original[:, 3] * diff[:, 1]
    t[:, 3] = original[:, 0] * diff[:, 3] - original[:, 1] * diff[:, 2] + original[:, 2] * diff[:, 1] + original[:, 3] * diff[:, 0]
    t = normalize_quaternion(t)
    return t


def main():
    import quaternion

    for j in range(100):

        q1 = torch.rand(7, 4)
        q1_norm = normalize_quaternion(q1)
        q1_inverse = inverse_quaternion(q1_norm)
        q1_conjugate = conjugate_quaternion(q1_norm)
        q1_norm_numpy = quaternion.as_quat_array(q1)
        identity = multiply_quaternion(q1_inverse, q1_norm)

        q2 = torch.rand(7, 4)
        q2_norm = normalize_quaternion(q2)
        q2_norm_numpy = quaternion.as_quat_array(q2)

        our_mult = multiply_quaternion(q2_norm, q1_norm)
        our_mult_optimized = multiply_quaternion_optimized(q2_norm, q1_norm)

        for i in range(len(q1)):
            q1_numpy_norm = q1_norm_numpy[i].normalized()
            q1_numpy_conjugate = q1_numpy_norm.conjugate()
            q1_numpy_inverse = q1_numpy_norm.inverse()

            q2_numpy_norm = q2_norm_numpy[i].normalized()
            mult = q2_numpy_norm * q1_numpy_norm

            pdb.set_trace()

            try:
                assert equality_quaternion(q1_numpy_norm, q1_norm[i]), 'Normalize does not work'
                assert equality_quaternion(q1_numpy_conjugate, q1_conjugate[i]), 'Conjugate does not work'
                assert equality_quaternion(q1_numpy_inverse, q1_inverse[i]), 'Inverse does not work'
                assert equality_quaternion(q1_numpy_inverse * q1_numpy_norm, identity[i]), 'Inverse does not work'
                assert equality_quaternion(mult, our_mult[i]), 'addition does not work'
                assert equality_quaternion(mult, our_mult_optimized[i]), 'addition optimized does not work'
                # assert sum(abs(quaternion.as_float_array(q2_numpy)-q2_norm[i])) < THR, 'Normalize does not work'
            except Exception as e:
                print(e)
                pdb.set_trace()
        print('Test is passed ', j)


if __name__ == "__main__":
    main()
