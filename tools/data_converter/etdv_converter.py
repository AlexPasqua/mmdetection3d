from pathlib import Path

from .etdv_data_utils import get_etdv_pc_info


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
    train_ptcld_ids = _read_imageset_file(str(imageset_folder / 'train.txt'))
    val_ptcld_ids = _read_imageset_file(str(imageset_folder / 'val.txt'))
    test_ptcld_ids = _read_imageset_file(str(imageset_folder / 'test.txt'))

    print('Generate info. this may take several minutes.')
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)

    etdv_infos_train = get_etdv_pc_info(
        data_path,
        training=True,
        pointcloud_ids=train_ptcld_ids,
        relative_path=relative_path)

    print(etdv_infos_train.keys())
    exit()

    _calculate_num_points_in_gt(data_path, etdv_infos_train, relative_path)
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'Kitti info train file is saved to {filename}')
    mmcv.dump(etdv_infos_train, filename)
    kitti_infos_val = get_etdv_pc_info(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        with_plane=with_plane,
        pointcloud_ids=val_ptcld_ids,
        relative_path=relative_path)
    _calculate_num_points_in_gt(data_path, kitti_infos_val, relative_path)
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'Kitti info val file is saved to {filename}')
    mmcv.dump(kitti_infos_val, filename)
    filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    print(f'Kitti info trainval file is saved to {filename}')
    mmcv.dump(etdv_infos_train + kitti_infos_val, filename)

    kitti_infos_test = get_etdv_pc_info(
        data_path,
        training=False,
        label_info=False,
        velodyne=True,
        calib=True,
        with_plane=False,
        pointcloud_ids=test_ptcld_ids,
        relative_path=relative_path)
    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'Kitti info test file is saved to {filename}')
    mmcv.dump(kitti_infos_test, filename)