"""
Training script for scene graph detection. Integrated with Rowan's faster rcnn setup
"""

from dataloaders.visual_genome import VGDataLoader, VG, build_graph_structure
import numpy as np
from torch import optim
import torch
import pandas as pd
import time
import os

from config import ModelConfig, BOX_SCALE, IM_SCALE
from torch.nn import functional as F
from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list, eval_entry
from lib.pytorch_misc import print_para
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pdb
# import KERN model
# from lib.kern_model import KERN

#--------updated--------
from lib.glat import GLATNET
import pdb
from torch.autograd import Variable
import copy
from scipy.special import softmax
import torch.optim.lr_scheduler as lr_scheduler

#--------updated--------
import sys
# import pickle
import _pickle as pickle
import os
codebase = '../../'
sys.path.append(codebase)
exp_name = 'motif'


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# conf = ModelConfig()
#--------updated--------
conf = ModelConfig()
if conf.model_s_m == 'motifnet':
    from lib.motifnet_model import RelModel
elif conf.model_s_m == 'stanford':
    from lib.stanford_model import RelModelStanford as RelModel
else:
    raise ValueError()


train, val, _ = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                          use_proposals=conf.use_proposals,
                          filter_non_overlap=conf.mode == 'sgdet')

ind_to_predicates = train.ind_to_predicates  # ind_to_predicates[0] means no relationship
ind_to_classes = train.ind_to_classes

print("conf.batch_size: ", conf.batch_size)

train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus)

# detector = KERN(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
#                 num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
#                 use_resnet=conf.use_resnet, use_proposals=conf.use_proposals, pooling_dim=conf.pooling_dim, return_top100=True)s

# python models/train_rels.py -m sgcls -model stanford -b 4 -p 400 -lr 1e-4 -ngpu 1 -ckpt checkpoints/vgdet/vg-24.tar -save_dir checkpoints/stanford -adam


order = 'leftright'
nl_obj = 2
nl_edge = 4
hidden_dim = 512
pass_in_obj_feats_to_decoder = False
pass_in_obj_feats_to_edge = False
rec_dropout = 0.1
use_bias = True
use_tanh = False
limit_vision = False



if conf.model_s_m == 'stanford':
    order = 'confidence'
    nl_edge = 2
    nl_obj = 1
    hidden_dim = 256
    rec_dropout = 0.1

    pass_in_obj_feats_to_decoder = False
    pass_in_obj_feats_to_edge = False
    use_bias = False
    use_tanh = False
    limit_vision = False
elif conf.model_s_m == 'motifnet':
    order = 'leftright'
    nl_obj = 2
    nl_edge = 4
    hidden_dim = 512
    pass_in_obj_feats_to_decoder = False
    pass_in_obj_feats_to_edge = False
    rec_dropout = 0.1
    use_bias = True
    use_tanh = False
    limit_vision = False


detector = RelModel(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                    num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                    use_resnet=conf.use_resnet, order=order,
                    nl_edge=nl_edge, nl_obj=nl_obj, hidden_dim=hidden_dim,
                    use_proposals=conf.use_proposals,
                    pass_in_obj_feats_to_decoder=pass_in_obj_feats_to_decoder,
                    pass_in_obj_feats_to_edge=pass_in_obj_feats_to_edge,
                    pooling_dim=conf.pooling_dim,
                    rec_dropout=rec_dropout,
                    use_bias=use_bias,
                    use_tanh=use_tanh,
                    limit_vision=limit_vision
                    )

#
#
# detector = RelModel(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
#                     num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
#                     use_resnet=conf.use_resnet, order=order,
#                     nl_edge=nl_edge, nl_obj=nl_obj, hidden_dim=hidden_dim,
#                     use_proposals=conf.use_proposals,
#                     pass_in_obj_feats_to_decoder=pass_in_obj_feats_to_decoder,
#                     pass_in_obj_feats_to_edge=pass_in_obj_feats_to_edge,
#                     pooling_dim=conf.pooling_dim,
#                     rec_dropout=rec_dropout,
#                     use_bias=use_bias,
#                     use_tanh=use_tanh,
#                     limit_vision=limit_vision,
#                     return_top100=True
#                     )


# Freeze all the motif model
for n, param in detector.named_parameters():
    param.requires_grad = False

print(print_para(detector), flush=True)

ckpt = torch.load(conf.ckpt)
print("Loading EVERYTHING from ckpt", conf.ckpt)
optimistic_restore(detector, ckpt['state_dict'])
detector.cuda()
start_epoch = -1


print('finish pretrained GLAT loading')


def train_epoch(epoch_num):
    detector.train()
    for n, param in detector.named_parameters():
        param.requires_grad = False
    # model.train()
    tr = []
    start = time.time()
    fpickle = open('./saved/{}_{}_{}_top{}_train.pkl'.format(conf.model_s_m, conf.mode, conf.batch_size, 100), 'wb')
    for b, batch in enumerate(train_loader):
        tr.append(train_batch(batch, fpickle, verbose=b % (conf.print_interval*10) == 0)) #b == 0))

        if b % conf.print_interval == 0 and b >= conf.print_interval:
            mn = pd.concat(tr[-conf.print_interval:], axis=1).mean(1)
            time_per_batch = (time.time() - start) / conf.print_interval
            print("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
                epoch_num, b, len(train_loader), time_per_batch, len(train_loader) * time_per_batch / 60))
            print(mn)
            print('-----------', flush=True)
            start = time.time()
    fpickle.close()

    return pd.concat(tr, axis=1)


def train_batch(b, fpickle, verbose=False):
    """
    :param b: contains:
          :param imgs: the image, [batch_size, 3, IM_SIZE, IM_SIZE]
          :param all_anchors: [num_anchors, 4] the boxes of all anchors that we'll be using
          :param all_anchor_inds: [num_anchors, 2] array of the indices into the concatenated
                                  RPN feature vector that give us all_anchors,
                                  each one (img_ind, fpn_idx)
          :param im_sizes: a [batch_size, 4] numpy array of (h, w, scale, num_good_anchors) for each image.

          :param num_anchors_per_img: int, number of anchors in total over the feature pyramid per img

          Training parameters:
          :param train_anchor_inds: a [num_train, 5] array of indices for the anchors that will
                                    be used to compute the training loss (img_ind, fpn_idx)
          :param gt_boxes: [num_gt, 4] GT boxes over the batch.
          :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
    :return:
    """
    # t0 = time.time()

    result, det_res = detector[b]

    # with open('./saved/{}_{}_top{}_train.pkl'.format(conf.model_s_m, conf.batch_size, 100), 'wb') as f:
    pickle.dump([result, det_res], fpickle)
    # print('1 train epoch saved')


    losses = {}
    if conf.use_ggnn_obj: # if not use ggnn obj, we just use scores of faster rcnn as their scores, there is no need to train
        losses['class_loss'] = F.cross_entropy(result.rm_obj_dists, result.rm_obj_labels)
    losses['rel_loss'] = F.cross_entropy(result.rel_dists, result.rel_labels[:, -1])
    loss = sum(losses.values())
    losses['total'] = loss
    res = pd.Series({x: y.data[0] for x, y in losses.items()})
    return res


def val_epoch():

    detector.eval()
    # model.eval()
    evaluator_list = [] # for calculating recall of each relationship except no relationship
    evaluator_multiple_preds_list = []
    fpickle = open('./saved/{}_{}_{}_top{}_eval.pkl'.format(conf.model_s_m, conf.mode, conf.batch_size, 100), 'wb')

    for index, name in enumerate(ind_to_predicates):
        if index == 0:
            continue
        evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))
        evaluator_multiple_preds_list.append((index, name, BasicSceneGraphEvaluator.all_modes(multiple_preds=True)))
    evaluator = BasicSceneGraphEvaluator.all_modes() # for calculating recall
    evaluator_multiple_preds = BasicSceneGraphEvaluator.all_modes(multiple_preds=True)
    # print(len(val_loader))
    for val_b, batch in enumerate(val_loader):
        # val_batch(conf.num_gpus * val_b, batch, evaluator, evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list)
        val_batch(val_b, batch, evaluator, evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list, fpickle)

    fpickle.close()
    print('eval saving finished ')


    recall = []
    recall_mp = []
    mean_recall = []
    mean_recall_mp = []
    return recall, recall_mp, mean_recall, mean_recall_mp


def rank_predicate(pred_entry):
    obj_scores0 = pred_entry['obj_scores'][pred_entry['pred_rel_inds'][:, 0]]
    obj_scores1 = pred_entry['obj_scores'][pred_entry['pred_rel_inds'][:, 1]]

    pred_scores_max = np.max(pred_entry['rel_scores'][:, 1:], axis=1)
    # pdb.set_trace()
    rel_scores_argmaxed = pred_scores_max * obj_scores0 * obj_scores1
    rel_scores_idx = np.argsort(rel_scores_argmaxed, axis=0)[::-1]

    pred_entry['rel_scores'] = pred_entry['rel_scores'][rel_scores_idx]
    pred_entry['pred_rel_inds'] = pred_entry['pred_rel_inds'][rel_scores_idx]

    return pred_entry


def val_batch(batch_num, b, evaluator, evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list, fpickle):
    # det_res = detector[b]
    dict_gt, det_res = detector[b]

    pickle.dump([dict_gt, det_res], fpickle)

    # if conf.num_gpus == 1:
    det_res = [det_res]


print("Training starts now!")
# optimizer = get_optim(conf.lr * conf.num_gpus * conf.batch_size)

for epoch in range(start_epoch + 1, start_epoch + 1 + conf.num_epochs):
    print("start training epoch: ", epoch)
    # scheduler.step()
    rez = train_epoch(epoch)
    print("overall{:2d}: ({:.3f})\n{}".format(epoch, rez.mean(1)['total'], rez.mean(1)), flush=True)

    recall, recall_mp, mean_recall, mean_recall_mp = val_epoch()
    break

print('finish all the saving')
