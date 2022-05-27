from concurrent import futures as futures
from pathlib import Path

from .kitti_data_utils import get_velodyne_path, get_label_path


def get_etdv_pc_info(path,
                     training=True,
                     label_info=True,
                     pointcloud_ids=350,
                     extend_matrix=True,
                     num_worker=8,
                     relative_path=True,
                     with_imageshape=True):
    """
    ETDV annotation format is a reduced version of KITTI annotation format.

    KITTI annotation format version 2:
    {
        [optional]points: [N, 3+] point cloud
        [optional, for kitti]image: {
            image_idx: ...
            image_path: ...
            image_shape: ...
        }
        point_cloud: {
            num_features: 4
            velodyne_path: ...
        }
        [optional, for kitti]calib: {
            R0_rect: ...
            Tr_velo_to_cam: ...
            P2: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: kitti difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """
    root_path = Path(path)
    if not isinstance(pointcloud_ids, list):
        pointcloud_ids = list(range(pointcloud_ids))

    def map_func(idx):
        info = {}
        pc_info = {'num_features': 4}
        annotations = None
        pc_info['velodyne_path'] = get_velodyne_path(
            idx, path, training, relative_path)
        
        print("\n\nHERE\n\n")
        print(f"pc_info: {pc_info}")
        exit()
        if label_info:
            label_path = get_label_path(idx, path, training, relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            
            print(pointcloud_ids)
            print(pc_info.keys())
            print(label_path)
            exit()

            annotations = get_label_anno(label_path)
        info['image'] = image_info
        info['point_cloud'] = pc_info
        if calib:
            calib_path = get_calib_path(
                idx, path, training, relative_path=False)
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]
                           ]).reshape([3, 4])
            P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]
                           ]).reshape([3, 4])
            P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]
                           ]).reshape([3, 4])
            P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]
                           ]).reshape([3, 4])
            if extend_matrix:
                P0 = _extend_matrix(P0)
                P1 = _extend_matrix(P1)
                P2 = _extend_matrix(P2)
                P3 = _extend_matrix(P3)
            R0_rect = np.array([
                float(info) for info in lines[4].split(' ')[1:10]
            ]).reshape([3, 3])
            if extend_matrix:
                rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                rect_4x4[3, 3] = 1.
                rect_4x4[:3, :3] = R0_rect
            else:
                rect_4x4 = R0_rect

            Tr_velo_to_cam = np.array([
                float(info) for info in lines[5].split(' ')[1:13]
            ]).reshape([3, 4])
            Tr_imu_to_velo = np.array([
                float(info) for info in lines[6].split(' ')[1:13]
            ]).reshape([3, 4])
            if extend_matrix:
                Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
                Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
            calib_info['P0'] = P0
            calib_info['P1'] = P1
            calib_info['P2'] = P2
            calib_info['P3'] = P3
            calib_info['R0_rect'] = rect_4x4
            calib_info['Tr_velo_to_cam'] = Tr_velo_to_cam
            calib_info['Tr_imu_to_velo'] = Tr_imu_to_velo
            info['calib'] = calib_info

        if with_plane:
            plane_path = get_plane_path(idx, path, training, relative_path)
            if relative_path:
                plane_path = str(root_path / plane_path)
            lines = mmcv.list_from_file(plane_path)
            info['plane'] = np.array([float(i) for i in lines[3].split()])

        if annotations is not None:
            info['annos'] = annotations
            add_difficulty_to_annos(info)
        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, pointcloud_ids)

    exit()
    return list(image_infos)