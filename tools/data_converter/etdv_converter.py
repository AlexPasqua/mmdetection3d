import mmcv
import numpy as np
from pathlib import Path

from .etdv_data_utils import get_etdv_pc_info
from mmdet3d.core.bbox import box_np_ops


# def _calculate_num_points_in_gt(infos):
#     for info in mmcv.track_iter_progress(infos):
#         annos = info['annos']
#         num_obj = len([n for n in annos['gt_names']])     # if n != 'DontCare'])
#         annos['num_points_in_gt'] = -np.ones(num_obj).astype(np.int32)


def _calculate_num_points_in_gt(data_path,
                                infos,
                                relative_path,
                                remove_outside=True,
                                num_features=4):
    for info in mmcv.track_iter_progress(infos):
        pc_info = info['lidar_points']
        # image_info = info['image']
        # calib = info['calib']
        if relative_path:
            v_path = str(Path(data_path) / pc_info['lidar_path'])
        else:
            v_path = pc_info['lidar_path']
        points_v = np.fromfile(
            v_path, dtype=np.float32, count=-1).reshape([-1, num_features])
        # rect = calib['R0_rect']
        # Trv2c = calib['Tr_velo_to_cam']
        # P2 = calib['P2']
        # if remove_outside:
        #     points_v = box_np_ops.remove_outside_points(
        #         points_v, rect, Trv2c, P2, image_info['image_shape'])

        # points_v = points_v[points_v[:, 0] > 0]
        annos = info['annos']
        num_obj = len([n for n in annos['gt_names'] if n != 'DontCare'])
        # annos = kitti.filter_kitti_anno(annos, ['DontCare'])
        dims = annos['dimensions'][:num_obj]
        loc = annos['location'][:num_obj]
        rots = annos['rotation_y'][:num_obj]
        gt_boxes_lidar = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
        # gt_boxes_lidar = box_np_ops.box_camera_to_lidar(
        #     gt_boxes_camera, rect, Trv2c)
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)

        # print('\n\n',np.where(indices),'\n\n')
        # exit()

        num_points_in_gt = indices.sum(0)
        num_ignored = len(annos['dimensions']) - num_obj
        num_points_in_gt = np.concatenate(
            [num_points_in_gt, -np.ones([num_ignored])])
        annos['num_points_in_gt'] = num_points_in_gt.astype(np.int32)


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line.strip('\n') for line in lines]


def create_etdv_info_file(data_path,
                           pkl_prefix='etdv',
                           save_path=None,
                           relative_path=True):
    """Create info file of ETDV dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str, optional): Prefix of the info file to be generated.
            Default: 'etdv'.
        save_path (str, optional): Path to save the info file.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
    """
    imageset_folder = Path(data_path) / 'ImageSets'
    train_pc_ids = _read_imageset_file(str(imageset_folder / 'train.txt'))
    val_pc_ids = _read_imageset_file(str(imageset_folder / 'val.txt'))
    test_pc_ids = _read_imageset_file(str(imageset_folder / 'test.txt'))

    print('Generate info. this may take several minutes.')
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)

    etdv_infos_train = get_etdv_pc_info(
        data_path,
        training=True,
        label_info=True,
        pc_ids=train_pc_ids,
        relative_path=relative_path
    )

    # it seems that the number of points is used only in db_sampler.py, for filterning ground truths
    # by number of points in the bbox.
    # We might skip this filtering and consider all the boxes, regardless of the number of points in them.
    # For now this func is a dummy that puts a dummy value as the number of points in each box:
        # this dummy value might be 0, 1, n, or -1 (-1 is currently used for boxes of class DontCare)
        # currently using -1
    _calculate_num_points_in_gt(data_path, etdv_infos_train, relative_path)

    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'ETDV info train file is saved to {filename}')
    mmcv.dump(etdv_infos_train, filename)
    
    etdv_infos_val = get_etdv_pc_info(
        data_path,
        training=True,
        label_info=True,
        pc_ids=val_pc_ids,
        relative_path=relative_path
    )
    _calculate_num_points_in_gt(data_path, etdv_infos_val, relative_path)
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'ETDV info val file is saved to {filename}')
    mmcv.dump(etdv_infos_val, filename)
    filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    print(f'ETDV info trainval file is saved to {filename}')
    mmcv.dump(etdv_infos_train + etdv_infos_val, filename)

    etdv_infos_test = get_etdv_pc_info(
        data_path,
        training=False,
        label_info=False,
        pc_ids=test_pc_ids,
        relative_path=relative_path
    )
    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'ETDV info test file is saved to {filename}')
    mmcv.dump(etdv_infos_test, filename)