from camera import get_camera_mat, get_random_pose, get_camera_pose
from camera_visualizer.visualize_cameras import visualize_cameras

if __name__ == "__main__":
    num_poses = 50
    H, W = 32,32
    camera_intrinsic = get_camera_mat(fov=10, res=(W, H))

    range_u, range_v = [0, 0], [0.4167, 0.5]
    # range_u, range_v = [0, 0.5], [0., 0.1]
    range_radius = [1., 1.]

    # range type defines whether to use range with np.random.uniform distribution (range) or np.random.choice (points).
    poses = get_random_pose(range_u, range_v, range_radius, range_u_type='points', range_v_type='range', batch_size=num_poses)
    

    cam_dict = {}
    for i in range(poses.shape[0]):
        cam_dict[f'camera_{i:0>3d}'] = {
            'K': camera_intrinsic.flatten().tolist(),
            'W2C': poses[i].inverse().flatten().tolist(),
            'img_size': [W, H]
        }
    sphere_radius = 1.
    camera_size = 0.1
    colored_camera_dicts = [
        ([0, 0, 1], cam_dict)
    ]

    visualize_cameras(colored_camera_dicts, sphere_radius, 
                      camera_size=camera_size, geometry_file=None)