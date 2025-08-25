import math
import torch
import numpy as np

_EPS = np.finfo(float).eps * 4.0


def matrix_to_quaternion(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(np.identity(4), True)
    >>> np.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(np.diag([1, -1, -1, 1]))
    >>> np.allclose(q, [0, 1, 0, 0]) or np.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> np.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> np.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> np.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True
    >>> R = euler_matrix(0.0, 0.0, np.pi/2.0)
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array(
            [
                [m00 - m11 - m22, 0.0, 0.0, 0.0],
                [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
            ]
        )
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


def quaternion_to_matrix(quaternion):
    """
    Return a homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> np.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> np.allclose(M, np.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> np.allclose(M, np.diag([1, -1, -1, 1]))
    True
    >>> M = quaternion_matrix([[1, 0, 0, 0],[0, 1, 0, 0]])
    >>> np.allclose(M, np.array([np.identity(4), np.diag([1, -1, -1, 1])]))
    True


    """
    q = np.array(quaternion, dtype=np.float64, copy=True).reshape((-1, 4))
    n = np.einsum("ij,ij->i", q, q)
    # how many entries do we have
    num_qs = len(n)
    identities = n < _EPS
    q[~identities, :] *= np.sqrt(2.0 / n[~identities, None])
    q = np.einsum("ij,ik->ikj", q, q)

    # store the result
    ret = np.zeros((num_qs, 4, 4))

    # pack the values into the result
    ret[:, 0, 0] = 1.0 - q[:, 2, 2] - q[:, 3, 3]
    ret[:, 0, 1] = q[:, 1, 2] - q[:, 3, 0]
    ret[:, 0, 2] = q[:, 1, 3] + q[:, 2, 0]
    ret[:, 1, 0] = q[:, 1, 2] + q[:, 3, 0]
    ret[:, 1, 1] = 1.0 - q[:, 1, 1] - q[:, 3, 3]
    ret[:, 1, 2] = q[:, 2, 3] - q[:, 1, 0]
    ret[:, 2, 0] = q[:, 1, 3] - q[:, 2, 0]
    ret[:, 2, 1] = q[:, 2, 3] + q[:, 1, 0]
    ret[:, 2, 2] = 1.0 - q[:, 1, 1] - q[:, 2, 2]
    ret[:, 3, 3] = 1.0
    # set any identities
    ret[identities] = np.eye(4)[None, ...]

    return ret.squeeze()


def matrix_to_euler_angles_np(R_w_i: np.array):
    """
    Convert rotation matrix to euler angles in Z-X-Y order
    """
    roll = np.arcsin(R_w_i[2, 1])
    pitch = np.arctan2(-R_w_i[2, 0] / np.cos(roll), R_w_i[2, 2] / np.cos(roll))
    yaw = np.arctan2(-R_w_i[0, 1] / np.cos(roll), R_w_i[1, 1] / np.cos(roll))

    return np.array([roll, pitch, yaw])


# the torch version of matrix_to_euler_angles, with batch support
def matrix_to_euler_angles_torch(R_w_i):
    """
    Convert rotation matrix to euler angles in Z-X-Y order
    :param R_w_i: rotation matrix, Float[Tensor, "B 3 3"]
    :return: euler angles, Float[Tensor, "B 3"]
    """
    roll = torch.asin(R_w_i[:, 2, 1])
    pitch = torch.atan2(-R_w_i[:, 2, 0] / torch.cos(roll), R_w_i[:, 2, 2] / torch.cos(roll))
    yaw = torch.atan2(-R_w_i[:, 0, 1] / torch.cos(roll), R_w_i[:, 1, 1] / torch.cos(roll))

    return torch.stack([roll, pitch, yaw], dim=1)


def make_rotation_by_up_and_eye(up: np.array, eye: np.array) -> np.array:
    """
    Create a rotation matrix from up and eye vectors
    :param up: up vector, np.array, "3", assumed has been normalized
    :param eye: eye vector, np.array, "3", assumed has been normalized
    :return: rotation matrix, np.array, "3 3"
    """
    new_c2w_R = np.eye(3)
    y = np.cross(up, eye)
    y = y / np.linalg.norm(y)
    z = np.cross(eye, y)
    z = z / np.linalg.norm(z)
    new_c2w_R[:, 0] = y
    new_c2w_R[:, 1] = eye
    new_c2w_R[:, 2] = z
    return new_c2w_R
