import sys
import os
import torch
import yaml

from easydict import EasyDict as edict
from pytorch_transformers.tokenization_bert import BertTokenizer
from vilbert.datasets import ConceptCapLoaderTrain, ConceptCapLoaderVal
from vilbert.vilbert import VILBertForVLTasks, BertConfig, BertForMultiModalPreTraining
from vilbert.task_utils import LoadDatasetEval

import numpy as np
import matplotlib.pyplot as plt
import PIL

from copy import deepcopy
import json
import random
import time
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from PIL import Image
# write arbitary string for given sentense. 
import _pickle as cPickle
import pickle
import cv2
import csv
import argparse
import glob
from types import SimpleNamespace
import ipdb
st = ipdb.set_trace

import pkbar

BATCH_SIZE = 1

class FeatureExtractor:
    MAX_SIZE = 1333
    MIN_SIZE = 800

    def __init__(self):
        self.args = self.get_parser()
        self.detection_model = self._build_detection_model()

    def get_parser(self):        
        parser = SimpleNamespace(model_file= 'data/detectron_model.pth',
                                config_file='data/detectron_config.yaml',
                                batch_size=1,
                                num_features=100,
                                feature_name="fc6",
                                confidence_threshold=0,
                                background=False,
                                partition=0)
        return parser
    
    def _build_detection_model(self):
        cfg.merge_from_file(self.args.config_file)
        cfg.freeze()

        model = build_detection_model(cfg)
        checkpoint = torch.load(self.args.model_file, map_location=torch.device("cpu"))

        load_state_dict(model, checkpoint.pop("model"))

        model.to("cuda")
        model.eval()
        return model

    def _image_transform(self, path):
        img = Image.open(path)
        im = np.array(img).astype(np.float32)
        # IndexError: too many indices for array, grayscale images
        if len(im.shape) < 3:
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
        im = im[:, :, ::-1]
        im -= np.array([102.9801, 115.9465, 122.7717])
        im_shape = im.shape
        im_height = im_shape[0]
        im_width = im_shape[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        # Scale based on minimum size
        im_scale = self.MIN_SIZE / im_size_min

        # Prevent the biggest axis from being more than max_size
        # If bigger, scale it down
        if np.round(im_scale * im_size_max) > self.MAX_SIZE:
            im_scale = self.MAX_SIZE / im_size_max

        im = cv2.resize(
            im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
        )
        img = torch.from_numpy(im).permute(2, 0, 1)

        im_info = {"width": im_width, "height": im_height}

        return img, im_scale, im_info

    def _process_feature_extraction(
        self, output, im_scales, im_infos, feature_name="fc6", conf_thresh=0
    ):
        batch_size = len(output[0]["proposals"])
        n_boxes_per_image = [len(boxes) for boxes in output[0]["proposals"]]
        score_list = output[0]["scores"].split(n_boxes_per_image)
        score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
        feats = output[0][feature_name].split(n_boxes_per_image)
        cur_device = score_list[0].device

        feat_list = []
        info_list = []

        for i in range(batch_size):
            dets = output[0]["proposals"][i].bbox / im_scales[i]
            scores = score_list[i]
            max_conf = torch.zeros((scores.shape[0])).to(cur_device)
            conf_thresh_tensor = torch.full_like(max_conf, conf_thresh)
            start_index = 1
            # Column 0 of the scores matrix is for the background class
            if self.args.background:
                start_index = 0
            for cls_ind in range(start_index, scores.shape[1]):
                cls_scores = scores[:, cls_ind]
                keep = nms(dets, cls_scores, 0.5)
                max_conf[keep] = torch.where(
                    # Better than max one till now and minimally greater than conf_thresh
                    (cls_scores[keep] > max_conf[keep])
                    & (cls_scores[keep] > conf_thresh_tensor[keep]),
                    cls_scores[keep],
                    max_conf[keep],
                )

            sorted_scores, sorted_indices = torch.sort(max_conf, descending=True)
            num_boxes = (sorted_scores[: self.args.num_features] != 0).sum()
            keep_boxes = sorted_indices[: self.args.num_features]
            feat_list.append(feats[i][keep_boxes])
            bbox = output[0]["proposals"][i][keep_boxes].bbox / im_scales[i]
            # Predict the class label using the scores
            objects = torch.argmax(scores[keep_boxes][start_index:], dim=1)
            cls_prob = torch.max(scores[keep_boxes][start_index:], dim=1)

            info_list.append(
                {
                    "bbox": bbox.cpu().numpy(),
                    "num_boxes": num_boxes.item(),
                    "objects": objects.cpu().numpy(),
                    "image_width": im_infos[i]["width"],
                    "image_height": im_infos[i]["height"],
                    "cls_prob": scores[keep_boxes].cpu().numpy(),
                }
            )

        return feat_list, info_list

    def get_detectron_features(self, image_paths):
        img_tensor, im_scales, im_infos = [], [], []

        for image_path in image_paths:
            im, im_scale, im_info = self._image_transform(image_path)
            img_tensor.append(im)
            im_scales.append(im_scale)
            im_infos.append(im_info)

        # Image dimensions should be divisible by 32, to allow convolutions
        # in detector to work
        current_img_list = to_image_list(img_tensor, size_divisible=32)
        current_img_list = current_img_list.to("cuda")

        with torch.no_grad():
            output = self.detection_model(current_img_list)

        feat_list = self._process_feature_extraction(
            output,
            im_scales,
            im_infos,
            self.args.feature_name,
            self.args.confidence_threshold,
        )

        return feat_list

    def _chunks(self, array, chunk_size):
        for i in range(0, len(array), chunk_size):
            yield array[i : i + chunk_size]

    def _save_feature(self, file_name, feature, info):
        file_base_name = os.path.basename(file_name)
        file_base_name = file_base_name.split(".")[0]
        info["image_id"] = file_base_name
        info["features"] = feature.cpu().numpy()
        file_base_name = file_base_name + ".npy"

        np.save(os.path.join(self.args.output_folder, file_base_name), info)

    def extract_features(self, image_paths):

        features, infos = self.get_detectron_features(image_paths)

        return features, infos

class IterDataset2d(Dataset):
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
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )

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
        pairs = self.get_pairs(anno)
        num_pairs = len(pairs)
        gt_boxes = self.get_object_rois(anno)
        obj_ids = self.get_object_ids(anno)
        obj_names = self.get_object_names(anno)
        preds = self.get_predicate_names(anno)
        queries = []
        gt_spatials = []
        input_masks = []
        segment_id_list = []
        og_sentences = []
        for idx in range(num_pairs):
            query = obj_names[pairs[idx][0]] + " " + preds[idx] + " " + obj_names[pairs[idx][1]]
            og_sentences.append(query)
            gt_spatials.append(gt_boxes[pairs[idx][0]])
            
            tokens = self.tokenizer.encode(query)
            tokens = self.tokenizer.add_special_tokens_single_sentence(tokens)

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            max_length = 37
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [0] * (max_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding
                
            queries.append(tokens)
            input_masks.append(input_mask)
            segment_id_list.append(segment_ids)
        
        
        return {
            "filename": anno['filename'],
            "file_path": os.path.join(self.im_path, anno['filename']),
            "num_queries": len(queries),
            # "scene": self._load_image(anno['filename']),
            "og_sentences": og_sentences,
            "object_boxes": torch.from_numpy(np.asarray(gt_spatials)).float(),
            "queries": torch.from_numpy(np.asarray(queries)),
            "input_masks": torch.from_numpy(np.asarray(input_masks)),
            "segment_ids": torch.from_numpy(np.asarray(segment_id_list))
        }

    def __len__(self):
        """Override __len__ method, return dataset's size."""
        return len(self.annos)

    def _load_image(self, img_name):
        """Load image and add augmentations."""
        img_name = os.path.join(self.im_path, img_name)
        _img = Image.open(img_name).convert('RGB')
        width, height = _img.size
        max_wh = max(width, height)
        preprocessing = transforms.Compose([
            transforms.Pad((0, 0, max_wh - width, max_wh - height)),
            transforms.ToTensor()
        ])
        return preprocessing(_img)

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
        return np.round(boxes[:, (2, 0, 3, 1)])

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

def tokenize_batch(batch):
    return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]

def untokenize_batch(batch):
    return [tokenizer.convert_ids_to_tokens(sent) for sent in batch]

def detokenize(sent):
    """ Roughly detokenizes (mainly undoes wordpiece) """
    new_sent = []
    for i, tok in enumerate(sent):
        if tok.startswith("##"):
            new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
        else:
            new_sent.append(tok)
    return new_sent

def printer(sent, should_detokenize=True):
    if should_detokenize:
        sent = detokenize(sent)[1:-1]
    print(" ".join(sent))


def process_feats(features, infos):
    num_image = len(infos)
    max_length = 37
    feature_list = []
    image_location_list = []
    image_mask_list = []
    for i in range(num_image):
        image_w = infos[i]['image_width']
        image_h = infos[i]['image_height']
        feature = features[i]
        num_boxes = feature.shape[0]

        g_feat = torch.sum(feature, dim=0) / num_boxes
        num_boxes = num_boxes + 1
        feature = torch.cat([g_feat.view(1,-1), feature], dim=0)
        boxes = infos[i]['bbox']
        image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
        image_location[:,:4] = boxes
        image_location[:,4] = (image_location[:,3] - image_location[:,1]) * (image_location[:,2] - image_location[:,0]) / (float(image_w) * float(image_h))
        image_location[:,0] = image_location[:,0] / float(image_w)
        image_location[:,1] = image_location[:,1] / float(image_h)
        image_location[:,2] = image_location[:,2] / float(image_w)
        image_location[:,3] = image_location[:,3] / float(image_h)
        g_location = np.array([0,0,1,1,1])
        image_location = np.concatenate([np.expand_dims(g_location, axis=0), image_location], axis=0)
        image_mask = [1] * (int(num_boxes))

        feature_list.append(feature)
        image_location_list.append(torch.tensor(image_location))
        image_mask_list.append(torch.tensor(image_mask))

    features = torch.stack(feature_list, dim=0).float().cuda()
    spatials = torch.stack(image_location_list, dim=0).float().cuda()
    image_mask = torch.stack(image_mask_list, dim=0).byte().cuda()
    co_attention_mask = torch.zeros((num_image, num_boxes, max_length)).cuda()
    return features, spatials, image_mask, co_attention_mask

def _iou2d(box_a, box_b):
    intersection = _intersect(box_a, box_b)
    vol_a = _area(box_a)
    vol_b = _area(box_b)
    union = vol_a + vol_b - intersection
    return intersection / union


def _intersect(box_a, box_b):
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])
    return max(0, xB - xA) * max(0, yB - yA)


def _area(box):
    return (box[2] - box[0]) * (box[3] - box[1])

def iter2d_collate_fn(batch):
    """Collate function for scene-aware ScanNet object classification."""
    total_objects = sum([ex["num_queries"] for ex in batch])
    return {
        "file_path": [ex["file_path"] for ex in batch],
        # "scene": torch.stack([ex["scene"] for ex in batch]),
        "filename": [ex["filename"] for ex in batch],
        "num_queries": [ex["num_queries"] for ex in batch],
        "og_sentences": [item for ex in batch for item in ex["og_sentences"]],
        "task": torch.ones((total_objects,1)) * 11,
        "object_boxes": torch.cat([ex["object_boxes"] for ex in batch]),
        "queries": torch.cat([ex["queries"] for ex in batch]),
        "input_masks": torch.cat([ex["input_masks"] for ex in batch]),
        "segment_ids": torch.cat([ex["segment_ids"] for ex in batch]),
    }

def load_gt_feats(filename):
    feat_dict = np.load(os.path.join("/projects/katefgroup/language_grounding/extra/vilbert_detectron_feats", filename[0][:-3] + 'npy'), allow_pickle=True)
    feat_dict = feat_dict.item()
    features = feat_dict.pop('features',None)
    try:
        assert features is not None
    except:
        print("None features")
        st()
        print(filename)
    infos = [feat_dict]
    features = torch.from_numpy(np.expand_dims(features,axis=0)).cuda()
    return features, infos

# =============================
# ViLBERT part
# =============================

args = SimpleNamespace(from_pretrained= "save/multitask_model/pytorch_model_9.bin",
                       bert_model="bert-base-uncased",
                       config_file="config/bert_base_6layer_6conect.json",
                       max_seq_length=101,
                       train_batch_size=1,
                       do_lower_case=True,
                       predict_feature=False,
                       seed=42,
                       num_workers=0,
                       baseline=False,
                       img_weight=1,
                       distributed=False,
                       objective=1,
                       visual_target=0,
                       dynamic_attention=False,
                       task_specific_tokens=True,
                       tasks='1',
                       save_name='',
                       in_memory=False,
                       batch_size=1,
                       local_rank=-1,
                       split='mteval',
                       clean_train_sets=True
                      )

config = BertConfig.from_json_file(args.config_file)
with open('./vilbert_tasks.yml', 'r') as f:
    task_cfg = edict(yaml.safe_load(f))

task_names = []
for i, task_id in enumerate(args.tasks.split('-')):
    task = 'TASK' + task_id
    name = task_cfg[task]['name']
    task_names.append(name)

timeStamp = args.from_pretrained.split('/')[-1] + '-' + args.save_name
config = BertConfig.from_json_file(args.config_file)
default_gpu=True

if args.predict_feature:
    config.v_target_size = 2048
    config.predict_feature = True
else:
    config.v_target_size = 1601
    config.predict_feature = False

if args.task_specific_tokens:
    config.task_specific_tokens = True    

if args.dynamic_attention:
    config.dynamic_attention = True

config.visualization = True
num_labels = 3129


model = VILBertForVLTasks.from_pretrained(
    args.from_pretrained, config=config, num_labels=num_labels, default_gpu=default_gpu
)
    
model.eval()
cuda = torch.cuda.is_available()
model = model.cuda()


dataloader = DataLoader(
            IterDataset2d(),
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=2,
            collate_fn = iter2d_collate_fn
        )


list1=['filename','query','pred_bbox_subject','iou']
with open("oracle_boxes.csv", "a") as fp:
    wr = csv.writer(fp, dialect='excel')
    wr.writerow(list1)

num_correct = 0
num_examples = 0

kbar = pkbar.Kbar(target=len(dataloader), width=25)
with torch.no_grad():
    for step, batch in enumerate(dataloader):
        if step>0:
            print(step, (num_correct/num_examples))
        features, infos = load_gt_feats(batch["filename"])
        features, spatials, image_mask, co_attention_mask = process_feats(features, infos)

        num_queries = torch.tensor(batch["num_queries"]).cuda()
        og_idx = np.repeat(np.arange(0, BATCH_SIZE), batch["num_queries"])
        features = torch.repeat_interleave(features, num_queries, dim=0)
        spatials = torch.repeat_interleave(spatials, num_queries, dim=0)
        image_mask = torch.repeat_interleave(image_mask, num_queries, dim=0)
        co_attention_mask = torch.repeat_interleave(co_attention_mask, num_queries, dim=0)
        try:
            _, _, _, _, _, _, vision_logit, _, _, _ = model(
                batch["queries"].cuda(), features, spatials, batch["segment_ids"].cuda(), batch["input_masks"].cuda(), image_mask, co_attention_mask, torch.Tensor(batch["task"]).long().cuda(), output_all_attention_masks=True
            )
            assert vision_logit.size()[0] == features.size()[0]
        except:
            print("Uggggghhhhh")
            print(step)
            print(batch["file_path"])
            print(vision_logit.size(), features.size())
            print(num_queries)
            st()
            print(features.size()[0])
            
        stats = []
        for curr_idx in range(features.size()[0]):
            height, width = infos[og_idx[curr_idx]]["image_height"], infos[og_idx[curr_idx]]["image_width"]

            _, grounding_idx = torch.sort(vision_logit[curr_idx].view(-1), 0, True)
            box = spatials[curr_idx][grounding_idx[0]][:4].tolist()
            y1 = int(box[1] * height)
            y2 = int(box[3] * height)
            x1 = int(box[0] * width)
            x2 = int(box[2] * width)
            pred_box = [x1, y1, x2, y2]
            calc_iou = _iou2d(batch["object_boxes"][curr_idx], pred_box)
            if calc_iou > 0.3:
                num_correct += 1
            num_examples += 1
            stats.append([batch["file_path"][og_idx[curr_idx]], batch["og_sentences"][curr_idx], pred_box, calc_iou])
        with open("oracle_boxes", "a") as f:
            writer = csv.writer(f)
            writer.writerows(stats)

print("Top 1 Accuracy: {}".format(num_correct/num_examples))