import numpy as np
import os
import json
import pickle
from copy import deepcopy
from tqdm import tqdm
import ipdb
st = ipdb.set_trace

class IterDataset2d():
    """Dataset utilities for conditional grounding."""

    def __init__(self, split='test'):
        """Initialize dataset."""
        super().__init__()
        self._path = '/projects/katefgroup/language_grounding/'
        self.im_path = self._path + 'VG/images/'
        self.anno_path = self._path + 'VG200/'
        self.split = split
        self.annos = self.load_annos()
        print('Loaded %d samples' % len(self.annos))

    def load_annos(self):
        """Load annotations."""
        with open(self.anno_path + 'VG200_preddet.json') as fid:
            annos = json.load(fid)
        with open (self.anno_path + 'vilbert_train_split', 'rb') as fp:
            allowed_files = set(pickle.load(fp))
        annos = [
            anno
            for anno in annos
            if (anno['filename'] not in allowed_files) and (anno['relations']['names'])
        ]
        return annos

    def __getitem__(self, index):
        """Get image's data (used by loader to later form a batch)."""
        anno = deepcopy(self.annos[index])
        gt_boxes = self.get_object_rois(anno)
        
        return {
            'file_name': anno['filename'],
            'file_path': os.path.join(self.im_path, anno['filename']),
            'bbox': gt_boxes,
            'num_box': len(gt_boxes)
        }

    def __len__(self):
        """Override __len__ method, return dataset's size."""
        return len(self.annos)

    
    @staticmethod
    def get_object_ids(anno):
        """Return object classes ids for given image."""
        return anno['objects']['ids']

    @staticmethod
    def get_object_names(anno):
        """Return object classes for given image."""
        return anno['objects']['names']

    @staticmethod
    def get_object_rois(anno):
        """Return rois for objects of given image."""
        boxes = np.array(anno['objects']['boxes'])
        boxes = np.round(boxes[:, (2, 0, 3, 1)])
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]
        boxes[:, 2] = width
        boxes[:, 3] = height
        return boxes

    @staticmethod
    def get_pairs(anno):
        """Return an array of related object ids for given image."""
        return np.stack((
            np.array(anno['relations']['subj_ids']),
            np.array(anno['relations']['obj_ids'])
        ), axis=1)

    @staticmethod
    def get_predicate_names(anno):
        """Return predicate classes for given image."""
        return anno['relations']['names']

if __name__=="__main__":
    vg_dataset = IterDataset2d()
    to_save = []
    for idx in tqdm(range(vg_dataset.__len__())):
        curr_sample = vg_dataset.__getitem__(idx)
        to_save.append(curr_sample)
    with open('vg200_boxes.npy', 'wb') as f:
        np.save(f, to_save)

