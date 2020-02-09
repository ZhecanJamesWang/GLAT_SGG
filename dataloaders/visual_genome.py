"""
File that involves dataloaders for the Visual Genome dataset.
"""

import json
import os

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from dataloaders.blob import Blob
from lib.fpn.box_intersections_cpu.bbox import bbox_overlaps
from config import ModelConfig, VG_IMAGES, IM_DATA_FN, VG_SGG_FN, VG_SGG_DICT_FN, BOX_SCALE, IM_SCALE, PROPOSAL_FN
from dataloaders.image_transforms import SquarePad, Grayscale, Brightness, Sharpness, Contrast, \
    RandomOrder, Hue, random_crop
from collections import defaultdict
from pycocotools.coco import COCO
import pdb
from tqdm import tqdm
import pickle
from torch.autograd import Variable

class VG(Dataset):
    def __init__(self, mode, roidb_file=VG_SGG_FN, dict_file=VG_SGG_DICT_FN,
                 image_file=IM_DATA_FN, filter_empty_rels=True, num_im=-1, num_val_im=5000,
                 filter_duplicate_rels=True, filter_non_overlap=True,
                 use_proposals=False):
        """
        Torch dataset for VisualGenome
        :param mode: Must be train, test, or val
        :param roidb_file:  HDF5 containing the GT boxes, classes, and relationships
        :param dict_file: JSON Contains mapping of classes/relationships to words
        :param image_file: HDF5 containing image filenames
        :param filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
        :param filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
        :param num_im: Number of images in the entire dataset. -1 for all images.
        :param num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        :param proposal_file: If None, we don't provide proposals. Otherwise file for where we get RPN
            proposals
        """
        if mode not in ('test', 'train', 'val'):
            raise ValueError("Mode must be in test, train, or val. Supplied {}".format(mode))
        self.mode = mode

        # Initialize
        self.roidb_file = roidb_file
        self.dict_file = dict_file
        self.image_file = image_file
        self.filter_non_overlap = filter_non_overlap
        self.filter_duplicate_rels = filter_duplicate_rels and self.mode == 'train'

        self.split_mask, self.gt_boxes, self.gt_classes, self.relationships = load_graphs(
            self.roidb_file, self.mode, num_im, num_val_im=num_val_im,
            filter_empty_rels=filter_empty_rels,
            filter_non_overlap=self.filter_non_overlap and self.is_train,
        )

        # pdb.set_trace()

        self.filenames = load_image_filenames(image_file)
        self.filenames = [self.filenames[i] for i in np.where(self.split_mask)[0]]

        self.ind_to_classes, self.ind_to_predicates = load_info(dict_file)

        if use_proposals:
            print("Loading proposals", flush=True)
            p_h5 = h5py.File(PROPOSAL_FN, 'r')
            rpn_rois = p_h5['rpn_rois']
            rpn_scores = p_h5['rpn_scores']
            rpn_im_to_roi_idx = np.array(p_h5['im_to_roi_idx'][self.split_mask])
            rpn_num_rois = np.array(p_h5['num_rois'][self.split_mask])

            self.rpn_rois = []
            for i in range(len(self.filenames)):
                rpn_i = np.column_stack((
                    rpn_scores[rpn_im_to_roi_idx[i]:rpn_im_to_roi_idx[i] + rpn_num_rois[i]],
                    rpn_rois[rpn_im_to_roi_idx[i]:rpn_im_to_roi_idx[i] + rpn_num_rois[i]],
                ))
                self.rpn_rois.append(rpn_i)
        else:
            self.rpn_rois = None

        # You could add data augmentation here. But we didn't.
        # tform = []
        # if self.is_train:
        #     tform.append(RandomOrder([
        #         Grayscale(),
        #         Brightness(),
        #         Contrast(),
        #         Sharpness(),
        #         Hue(),
        #     ]))

        tform = [
            SquarePad(),
            Resize(IM_SCALE),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.transform_pipeline = Compose(tform)

    @property
    def coco(self):
        """
        :return: a Coco-like object that we can use to evaluate detection!
        """
        anns = []
        for i, (cls_array, box_array) in enumerate(zip(self.gt_classes, self.gt_boxes)):
            for cls, box in zip(cls_array.tolist(), box_array.tolist()):
                anns.append({
                    'area': (box[3] - box[1] + 1) * (box[2] - box[0] + 1),
                    'bbox': [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1],
                    'category_id': cls,
                    'id': len(anns),
                    'image_id': i,
                    'iscrowd': 0,
                })
        fauxcoco = COCO()
        fauxcoco.dataset = {
            'info': {'description': 'ayy lmao'},
            'images': [{'id': i} for i in range(self.__len__())],
            'categories': [{'supercategory': 'person',
                               'id': i, 'name': name} for i, name in enumerate(self.ind_to_classes) if name != '__background__'],
            'annotations': anns,
        }
        fauxcoco.createIndex()
        return fauxcoco

    @property
    def is_train(self):
        return self.mode.startswith('train')

    @classmethod
    def splits(cls, *args, **kwargs):
        """ Helper method to generate splits of the dataset"""
        train = cls('train', *args, **kwargs)
        val = cls('val', *args, **kwargs)
        test = cls('test', *args, **kwargs)
        return train, val, test

    def __getitem__(self, index):
        image_unpadded = Image.open(self.filenames[index]).convert('RGB')

        # Optionally flip the image if we're doing training
        flipped = self.is_train and np.random.random() > 0.5
        gt_boxes = self.gt_boxes[index].copy()

        # Boxes are already at BOX_SCALE
        if self.is_train:
            # crop boxes that are too large. This seems to be only a problem for image heights, but whatevs
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]].clip(
                None, BOX_SCALE / max(image_unpadded.size) * image_unpadded.size[1])
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]].clip(
                None, BOX_SCALE / max(image_unpadded.size) * image_unpadded.size[0])

            # # crop the image for data augmentation
            # image_unpadded, gt_boxes = random_crop(image_unpadded, gt_boxes, BOX_SCALE, round_boxes=True)

        w, h = image_unpadded.size
        box_scale_factor = BOX_SCALE / max(w, h)

        if flipped:
            scaled_w = int(box_scale_factor * float(w))
            # print("Scaled w is {}".format(scaled_w))
            image_unpadded = image_unpadded.transpose(Image.FLIP_LEFT_RIGHT)
            gt_boxes[:, [0, 2]] = scaled_w - gt_boxes[:, [2, 0]]

        img_scale_factor = IM_SCALE / max(w, h)
        if h > w:
            im_size = (IM_SCALE, int(w * img_scale_factor), img_scale_factor)
        elif h < w:
            im_size = (int(h * img_scale_factor), IM_SCALE, img_scale_factor)
        else:
            im_size = (IM_SCALE, IM_SCALE, img_scale_factor)

        gt_rels = self.relationships[index].copy()
        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.mode == 'train'
            old_size = gt_rels.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in gt_rels:
                all_rel_sets[(o0, o1)].append(r)
            gt_rels = [(k[0], k[1], np.random.choice(v)) for k,v in all_rel_sets.items()]
            gt_rels = np.array(gt_rels)


        entry = {
            'img': self.transform_pipeline(image_unpadded),
            'img_size': im_size,
            'gt_boxes': gt_boxes,
            'gt_classes': self.gt_classes[index].copy(),
            'gt_relations': gt_rels,
            'scale': IM_SCALE / BOX_SCALE,  # Multiply the boxes by this.
            'index': index,
            'flipped': flipped,
            'image_id': self.filenames[index],
        }

        if self.rpn_rois is not None:
            entry['proposals'] = self.rpn_rois[index]

        assertion_checks(entry)
        return entry

    def __len__(self):
        return len(self.filenames)

    @property
    def num_predicates(self):
        return len(self.ind_to_predicates)

    @property
    def num_classes(self):
        return len(self.ind_to_classes)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MISC. HELPER FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def assertion_checks(entry):
    im_size = tuple(entry['img'].size())
    if len(im_size) != 3:
        raise ValueError("Img must be dim-3")

    c, h, w = entry['img'].size()
    if c != 3:
        raise ValueError("Must have 3 color channels")

    num_gt = entry['gt_boxes'].shape[0]
    if entry['gt_classes'].shape[0] != num_gt:
        raise ValueError("GT classes and GT boxes must have same number of examples")

    assert (entry['gt_boxes'][:, 2] >= entry['gt_boxes'][:, 0]).all()
    assert (entry['gt_boxes'] >= -1).all()


def load_image_filenames(image_file, image_dir=VG_IMAGES):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    :param image_file: JSON file. Elements contain the param "image_id".
    :param image_dir: directory where the VisualGenome images are located
    :return: List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue

        filename = os.path.join(image_dir, basename)
        if os.path.exists(filename):
            fns.append(filename)
    assert len(fns) == 108073
    return fns


def load_graphs(graphs_file, mode='train', num_im=-1, num_val_im=0, filter_empty_rels=True,
                filter_non_overlap=False):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    :param graphs_file: HDF5
    :param mode: (train, val, or test)
    :param num_im: Number of images we want
    :param num_val_im: Number of validation images
    :param filter_empty_rels: (will be filtered otherwise.)
    :param filter_non_overlap: If training, filter images that dont overlap.
    :return: image_index: numpy array corresponding to the index of images we're using
             boxes: List where each element is a [num_gt, 4] array of ground 
                    truth boxes (x1, y1, x2, y2)
             gt_classes: List where each element is a [num_gt] array of classes
             relationships: List where each element is a [num_r, 3] array of 
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    if mode not in ('train', 'val', 'test'):
        raise ValueError('{} invalid'.format(mode))

    roi_h5 = h5py.File(graphs_file, 'r')
    data_split = roi_h5['split'][:]
    split = 2 if mode == 'test' else 0
    split_mask = data_split == split

    # Filter out images without bounding boxes
    split_mask &= roi_h5['img_to_first_box'][:] >= 0
    if filter_empty_rels:
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0

    image_index = np.where(split_mask)[0]
    if num_im > -1:
        image_index = image_index[:num_im]
    if num_val_im > 0:
        if mode == 'val':
            image_index = image_index[:num_val_im]
        elif mode == 'train':
            image_index = image_index[num_val_im:]


    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True

    # Get box information
    all_labels = roi_h5['labels'][:, 0]

    all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # will index later
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]


    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    boxes = []
    gt_classes = []
    relationships = []
    for i in range(len(image_index)):
        boxes_i = all_boxes[im_to_first_box[i]:im_to_last_box[i] + 1, :]
        gt_classes_i = all_labels[im_to_first_box[i]:im_to_last_box[i] + 1]

        if im_to_first_rel[i] >= 0:
            predicates = _relation_predicates[im_to_first_rel[i]:im_to_last_rel[i] + 1]
            obj_idx = _relations[im_to_first_rel[i]:im_to_last_rel[i] + 1] - im_to_first_box[i]
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            rels = np.column_stack((obj_idx, predicates))
        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)

        if filter_non_overlap:
            assert mode == 'train'
            inters = bbox_overlaps(boxes_i, boxes_i)
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        relationships.append(rels)

    return split_mask, boxes, gt_classes, relationships


def load_info(info_file):
    """
    Loads the file containing the visual genome label meanings
    :param info_file: JSON
    :return: ind_to_classes: sorted list of classes
             ind_to_predicates: sorted list of predicates
    """
    info = json.load(open(info_file, 'r'))
    info['label_to_idx']['__background__'] = 0
    info['predicate_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])

    return ind_to_classes, ind_to_predicates


def vg_collate(data, num_gpus=3, is_train=False, mode='det'):
    assert mode in ('det', 'rel')
    blob = Blob(mode=mode, is_train=is_train, num_gpus=num_gpus,
                batch_size_per_gpu=len(data) // num_gpus)
    for d in data:
        blob.append(d)
    blob.reduce()
    return blob


class VGDataLoader(torch.utils.data.DataLoader):
    """
    Iterates through the data, filtering out None,
     but also loads everything as a (cuda) variable
    """

    @classmethod
    def splits(cls, train_data, val_data, batch_size=3, num_workers=1, num_gpus=3, mode='det',
               **kwargs):
        assert mode in ('det', 'rel')
        train_load = cls(
            dataset=train_data,
            batch_size=batch_size * num_gpus,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=lambda x: vg_collate(x, mode=mode, num_gpus=num_gpus, is_train=True),
            drop_last=True,
            # pin_memory=True,
            **kwargs,
        )
        val_load = cls(
            dataset=val_data,
            batch_size=batch_size * num_gpus if mode=='det' else num_gpus,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: vg_collate(x, mode=mode, num_gpus=num_gpus, is_train=False),
            drop_last=True,
            # pin_memory=True,
            **kwargs,
        )
        return train_load, val_load

def build_graph_structure(entries, index2name_object, index2name_predicate, if_predicting=False, sgclsdet=False):
    # Input: pred_relations(Tensor) pred_classes(Variable)
    # Output: adj(Tensor) node_class(Variable) nodes_type(Tensor)
    total_data = {}
    total_data['adj'] = []
    # total_data['node_name'] = []
    total_data['node_class'] = []
    # total_data['img_id'] = []
    total_data['node_type'] = []

    entries_minibatch = {}
    entries_minibatch['pred_relations'] = []
    entries_minibatch['pred_classes'] = []

    # For SGCLS
    useless_entity_id = []
    start_id = 0

    if entries['pred_relations'].size(1) == 4:
        # pdb.set_trace()
        # entries_minibatch['pred_relations'].append(entries['pred_relations'][:, 1:])
        # entries_minibatch['pred_classes'].append(entries['pred_classes'])
        for i in range(entries['pred_relations'][:, 0].max()+1):
            rel_idx_cur_img = (entries['pred_relations'][:, 0] == i).view(-1, 1).expand(-1, 4)
            entries_minibatch['pred_relations'].append(entries['pred_relations'][rel_idx_cur_img].view(-1, 4)[:, 1:])
            entity_idx_cur_img = entries_minibatch['pred_relations'][i][:, :2]
            entries_minibatch['pred_classes'].append(entries['pred_classes'][entity_idx_cur_img.min():entity_idx_cur_img.max()+1])

            # For SGCLS
            end_id = entity_idx_cur_img.min()
            # pdb.set_trace()
            if start_id != end_id:
                useless_entity_id += list(range(start_id, end_id))
            start_id = entity_idx_cur_img.max() + 1
            if i == entries['pred_relations'][:, 0].max() and start_id != entries['pred_classes'].size(0):
                useless_entity_id += list(range(start_id, entries['pred_classes'].size(0)))

            # pdb.set_trace()
            entries_minibatch['pred_relations'][i][:, :2] = entries_minibatch['pred_relations'][i][:, :2] - entries_minibatch['pred_relations'][i][:, :2].min()

            # print('go through 460 in visual_genome.py')
            # pdb.set_trace()
    else:
        entries_minibatch['pred_relations'].append(entries['pred_relations'])
        entries_minibatch['pred_classes'].append(entries['pred_classes'])

    for i in range(len(entries_minibatch['pred_classes'])):
        # if if_predicting:
        #     return_classes = entry['pred_classes']
        #     return_relations = entry['pred_relations']
        # else:
        #     return_classes = entry['gt_classes']
        #     return_relations = entry['gt_relations']
        return_classes = entries_minibatch['pred_classes'][i]
        return_relations = entries_minibatch['pred_relations'][i]
        entity_num = return_classes.size(0)
        total_node_num = entity_num + return_relations.size(0)
        # pdb.set_trace()
        nodes_class = torch.cat((return_classes, Variable(return_relations[:, -1])), dim=0)
        # pdb.set_trace()
        nodes_type = torch.ones_like(return_classes).data
        nodes_type = torch.cat((nodes_type, torch.zeros_like(return_relations[:,0])), dim=0)
        adj = torch.zeros(total_node_num, total_node_num).type_as(return_relations)
        # pdb.set_trace()
        # print('position', i)
        # print('------adj size', adj.size(), '------relation size',return_relations.size(), )
        # nodes_name = [] + [index2name_object[i] for i in return_classes.tolist()]
        for j, relation in enumerate(return_relations.tolist()):
            # nodes_name.append(index2name_predicate[relation[-1]])
            try:
                adj[relation[0]][entity_num+j] = 1
            except Exception as e:
                print(e)
                pdb.set_trace()
            # print('position', )
            adj[entity_num+j][relation[1]] = 2
        total_data['adj'].append(adj)
        # total_data['node_name'].append(nodes_name)
        total_data['node_class'].append(nodes_class)
        total_data['node_type'].append(nodes_type)

        # total_node_num = len(return_classes) + return_relations.shape[0]
        # nodes_class = [] + list(return_classes)
        # nodes_name = [] + [index2name_object[i] for i in list(return_classes)]
        # nodes_type = [] + len(return_classes) * [1]   # entity:1 predicate:0
        # adj = torch.zeros(total_node_num, total_node_num)
        # entity_num = len(list(return_classes))
        # for j, relation in enumerate(return_relations.tolist()):
        #     nodes_class.append(relation[-1])
        #     nodes_name.append(index2name_predicate[relation[-1]])
        #     nodes_type.append(0)
        #     adj[relation[0]][entity_num+j] = 1
        #     adj[entity_num+j][relation[1]] = 2
        # total_data['adj'].append(adj)
        # total_data['node_name'].append(np.asarray(nodes_name))
        # total_data['node_class'].append(np.asarray(nodes_class))
        # # total_data['img_id'].append(entry['image_id'])
        # total_data['node_type'].append(np.asarray(nodes_type))
        # pdb.set_trace()
    if not sgclsdet:
        return total_data
    else:
        return total_data, useless_entity_id


# def build_graph_structure_reverse(entries, pred_label_predicate):
    # total_data['adj']
    # total_data['node_name']
    # total_data['node_class']
    # total_data['img_id']
    # total_data['node_type']
    # new_entries = {}
    # new_entries['gt_relations'] = []
    # new_entries['gt_classes'] = []
    # new_entries['gt_boxes'] = []
    #
    # for i, entry in enumerate(entries):
    #     node_type = total_data['node_type'][i]
    #     node_class = total_data['node_class'][i]
    #     adj = total_data['adj'][i]
    #
    #     gt_boxes = entry['gt_boxes']
    #
    #     new_gt_relations = node_class[node_type == 0]
    #     new_gt_classes = node_class[node_type == 1]
    #
    #     new_entries['gt_classes'].append(new_gt_classes)
    #     new_entries['gt_classes'].append(gt_boxes)
    #
    #     return_gt_relations = []
    #
    #     for i in range(len(new_gt_relations)):
    #         new_relation = new_gt_relations[i]
    #         new_entity = new_gt_classes[i]
    #
    #         relation = pred_label_predicate[i]
    #
    #         new_relation = np.asarray([new_entity[relation[0]], new_entity[relation[1]], new_relation])
    #         return_gt_relations.append(new_relation)
    #
    #     new_entries['gt_relations'].append(return_gt_relations)
    #
    # new_entries['gt_relations'] = np.asarray(new_entries['gt_relations'])
    # new_entries['gt_classes'] = np.asarray(new_entries['gt_classes'])
    # new_entries['gt_boxes'] = np.asarray(new_entries['gt_boxes'])

    # pred_entry = {
    #     'pred_boxes': boxes_i * BOX_SCALE / IM_SCALE,  # (23, 4) (16, 4)
    #     'pred_classes': objs_i,  # (23,) (16,)
    #     'pred_rel_inds': rels_i,  # (506, 2) (240, 2)
    #     'obj_scores': obj_scores_i,  # (23,) (16,)
    #     'rel_scores': pred_scores_i,  # hack for now. (506, 51) (240, 51)
    # }

    # entries['rel_scores'] = pred_label_predicate

    # return entries


if __name__=='__main__':
    conf = ModelConfig()
    train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                              use_proposals=conf.use_proposals,
                              filter_non_overlap=conf.mode == 'sgdet')
    train_entities_list = []
    train_num = 20000
    val_num = 1000
    test_num = 5000

    if train_num == 'full':
        train_num = len(train)
        val_num = len(val)
        test_num = len(test)

    for i in tqdm(range(train_num)):
        train_entities_list.append(train.__getitem__(i))

    val_entities_list = []
    for i in tqdm(range(val_num)):
        val_entities_list.append(val.__getitem__(i))

    test_entities_list = []
    for i in tqdm(range(test_num)):
        test_entities_list.append(test.__getitem__(i))

    train_data = build_graph_structure(train_entities_list, train.ind_to_classes, train.ind_to_predicates)
    val_data = build_graph_structure(val_entities_list, val.ind_to_classes, val.ind_to_predicates)
    test_data = build_graph_structure(test_entities_list, test.ind_to_classes, test.ind_to_predicates)

    save_root = '/home/haoxuan/code/KERN/data/'

    filename = os.path.join(save_root, 'train_VG_kern_{}.pkl'.format('_'.join([str(train_num), str(val_num), str(test_num)])))
    with open(filename, 'wb') as f:
        pickle.dump(train_data, f)

    filename = os.path.join(save_root, 'eval_VG_kern_{}.pkl'.format('_'.join([str(train_num), str(val_num), str(test_num)])))
    with open(filename, 'wb') as f:
        pickle.dump(val_data, f)

    filename = os.path.join(save_root, 'test_VG_kern_{}.pkl'.format('_'.join([str(train_num), str(val_num), str(test_num)])))
    with open(filename, 'wb') as f:
        pickle.dump(test_data, f)

    filename = os.path.join(save_root, 'ind_to_classes_{}.pkl'.format('_'.join([str(train_num), str(val_num), str(test_num)])))
    with open(filename, 'wb') as f:
        pickle.dump(train.ind_to_classes, f)

    filename = os.path.join(save_root, 'ind_to_predicates_{}.pkl'.format('_'.join([str(train_num), str(val_num), str(test_num)])))
    with open(filename, 'wb') as f:
        pickle.dump(train.ind_to_predicates, f)
