import mmcv
import numpy as np
from pathlib import Path

from .etdv_data_utils import get_etdv_pc_info


def _calculate_num_points_in_gt(infos):
    for info in mmcv.track_iter_progress(infos):
        annos = info['annos']
        num_obj = len([n for n in annos['name']])     # if n != 'DontCare'])
        annos['num_points_in_gt'] = -np.ones(num_obj).astype(np.int32)


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
    _calculate_num_points_in_gt(etdv_infos_train)

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
    _calculate_num_points_in_gt(etdv_infos_val)
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