import numpy as np
import trimesh

colors = np.array([
    [255, 0, 0, 255],    # 红色 (Red)
    [0, 255, 0, 255],    # 绿色 (Green)
    [0, 0, 255, 255],    # 蓝色 (Blue)
    [255, 255, 0, 255],  # 黄色 (Yellow)
    [255, 165, 0, 255],  # 橙色 (Orange)
    [128, 0, 128, 255],  # 紫色 (Purple)
    [0, 255, 255, 255],  # 青色 (Cyan)
    [255, 192, 203, 255],# 粉色 (Pink)
    [128, 128, 128, 255],# 灰色 (Gray)
    [0, 0, 0, 255]       # 黑色 (Black)
], dtype=np.uint8)


def visualize_kp(groups, log):
    log(f'kp_shape: {groups[0].shape} len: {len(groups)}')
    point_cloud = []
    for i in range(min(10, len(groups))):
        points = groups[i].reshape(-1, 3).detach().cpu().numpy()
        point_cloud.append(trimesh.points.PointCloud(points, colors=colors[i]))

    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    scene = trimesh.Scene([box] + point_cloud)

    if scene.export("output.glb"):
        log("GLB file saved as output.glb")
    else:
        log("Failed to save GLB file!")
