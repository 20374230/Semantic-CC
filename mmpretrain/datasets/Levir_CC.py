# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import List

import mmengine
from mmengine.dataset import BaseDataset
from mmengine.fileio import get_file_backend
from .basecddataset import _BaseCDDataset
from mmpretrain.registry import DATASETS
import copy
import os.path as osp
from typing import Callable, Dict, List, Optional, Sequence, Union

import mmengine
import mmengine.fileio as fileio
import numpy as np
from mmengine.dataset import BaseDataset, Compose

from mmpretrain.registry import DATASETS

@DATASETS.register_module()
class LevirCCcaptions(BaseDataset):
    """COCO Caption dataset.

    Args:
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``..
        ann_file (str): Annotation file path.
        data_prefix (dict): Prefix for data field. Defaults to
            ``dict(img_path='')``.
        pipeline (Sequence): Processing pipeline. Defaults to an empty tuple.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        #img_prefix = self.data_prefix['img_path']
        img_dir_from = self.data_prefix.get('img_path_from', None)
        img_dir_to = self.data_prefix.get('img_path_to', None)
        annotations = mmengine.load(self.ann_file)
       # file_backend = get_file_backend(img_prefix)

        data_list = []
        for ann in annotations['images']:
            #
            # print(self.data_root+ann['filepath'])
            # print(self.data_prefix['type'])
            if not(self.data_root+ann['filepath']==self.data_prefix['type']):
                continue
            img_name=ann['filename']
            #print(ann)
           
            for gt in ann['sentences']:
                #for tt in gt['raw']:
                gt_caption=[]
                gt_caption.append( gt['raw'])
                #print
                #print(gt['raw'])
                data_info = dict(image_id= ann['imgid']
                                ,img_path=\
                                    [osp.join(img_dir_from, img_name ), \
                                    osp.join(img_dir_to, img_name)],
                                    gt_caption= gt_caption)
               # print(data_info)
                data_list.append(data_info)
        #print(data_list)
        return data_list

@DATASETS.register_module()
class LevirCCcaptions2(_BaseCDDataset):
    """COCO Caption dataset.

    Args:
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``..
        ann_file (str): Annotation file path.
        data_prefix (dict): Prefix for data field. Defaults to
            ``dict(img_path='')``.
        pipeline (Sequence): Processing pipeline. Defaults to an empty tuple.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """
    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 format_seg_map='to_binary',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            format_seg_map=format_seg_map,
            **kwargs)
    def load_data_list(self) -> List[dict]:
        """Load data list."""
        data_list = []
        img_dir_from = self.data_prefix.get('img_path_from', None)
        img_dir_to = self.data_prefix.get('img_path_to', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        #img_prefix = self.data_prefix['img_path']
        # img_dir_from = self.data_prefix.get('img_path_from', None)
        # img_dir_to = self.data_prefix.get('img_path_to', None)
        annotations = mmengine.load(self.ann_file)
       # file_backend = get_file_backend(img_prefix)

        data_list = []
        for ann in annotations['images']:
            #
            # print(self.data_root+ann['filepath'])
            # print(self.data_prefix['type'])
            if not(self.data_root+ann['filepath']==self.data_prefix['type']):
                continue
            img_name=ann['filename']
            #print(ann)
           
            for gt in ann['sentences']:
                #for tt in gt['raw']:
                gt_caption=[]
                gt_caption.append( gt['raw'])
                #print
                #print(gt['raw'])
                data_info = dict(image_id= ann['imgid']
                                ,img_path=\
                                    [osp.join(img_dir_from, img_name ), \
                                    osp.join(img_dir_to, img_name)],
                                    gt_caption= gt_caption,
                                    
                                    )
               # print(data_info)
                if ann_dir is not None:
                    seg_map = img_name.replace(self.img_suffix, self.seg_map_suffix)
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['format_seg_map'] = self.format_seg_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        #print(data_list)
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        return data_list