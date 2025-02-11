import torch
import math
import json
import numpy as np

def spiral(radius=1):
    return lambda theta, phi : [
                                radius*np.sin(np.arccos(1 - 2*(np.clip(phi, 1e-5, math.pi - 1e-5)/ math.pi))) * np.sin(math.pi-theta), # 1
                                radius*np.cos(np.arccos(1 - 2*(np.clip(phi, 1e-5, math.pi - 1e-5)/ math.pi))), # 2
                                radius * np.sin(np.arccos(1 - 2*(np.clip(phi, 1e-5, math.pi - 1e-5)/ math.pi))) * np.cos(math.pi-theta), # 3
                                ]

def look_at_rotation(camera_position, at=None, up=None, inverse=False, cv=False):
    """
    This function takes a vector 'camera_position' which specifies the location
    of the camera in world coordinates and two vectors `at` and `up` which
    indicate the position of the object and the up directions of the world
    coordinate system respectively. The object is assumed to be centered at
    the origin.
    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.
    Input:
        camera_position: 3
        at: 1 x 3 or N x 3  (0, 0, 0) in default
        up: 1 x 3 or N x 3  (0, 1, 0) in default
    """

    if at is None:
        at = torch.zeros_like(camera_position)
    else:
        at = torch.tensor(at).type_as(camera_position)
    if up is None:
        up = torch.zeros_like(camera_position)
        up[2] = -1
    else:
        up = torch.tensor(up).type_as(camera_position)

    z_axis = normalize(at - camera_position)[0]
    x_axis = normalize(cross(z_axis, up, -1))[0]
    y_axis = normalize(cross(x_axis, z_axis, -1))[0]

    R = cat([x_axis[:, None], y_axis[:, None], z_axis[:, None]], axis=1)
    return R

def normalize(x, axis=-1, order=2):
    if isinstance(x, torch.Tensor):
        l2 = x.norm(p=order, dim=axis, keepdim=True)
        return x / (l2 + 1e-8), l2

    else:
        l2 = np.linalg.norm(x, order, axis)
        l2 = np.expand_dims(l2, axis)
        l2[l2 == 0] = 1
        return x / l2,

def cat(x, axis=1):
    if isinstance(x[0], torch.Tensor):
        return torch.cat(x, dim=axis)
    return np.concatenate(x, axis=axis)

def cross(x, y, axis=0):
    T = torch if isinstance(x, torch.Tensor) else np
    return T.cross(x, y, axis)

if __name__ == '__main__':
    azimuths = [0.0, 4.04, 15.47, 33.22, 56.25, 83.5, 113.91, 146.43, 180, 213.57, 246.09, 276.5, 303.75, 326.78,
                344.53, 355.96]
    elevations = [-10, 0, 10, 20, 30, 40]
    cameras = []
    at = (0, 0, 0)
    up = (0, -1, 0)

    i = 0
    pose_gen = spiral()
    for elevation in elevations:
        for azimuth in azimuths:
            theta = azimuth * math.pi / 180
            phi = (90 - elevation) * math.pi / 180

            c2w = torch.eye(4)
            cam_pos = torch.tensor(pose_gen(theta, phi))
            cam_rot = look_at_rotation(cam_pos, at=at, up=up, inverse=False, cv=True)
            c2w[:3, 3], c2w[:3, :3] = cam_pos, cam_rot

            cameras.append(
                {
                    "name": f'{i :04d}.png',
                    "extrinsics": c2w.detach().numpy().tolist()
                }
            )
            i += 1

    output_file = "/home/wcc/RodinHD/data/metahuman_data/parameters.json"
    with open(output_file, "w") as f:
        json.dump(cameras, f, indent=4)