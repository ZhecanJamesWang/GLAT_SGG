# create predictions from the other stuff
"""
Go from proposals + scores to relationships.

pred-cls: No bbox regression, obj dist is exactly known
sg-cls : No bbox regression
sg-det : Bbox regression

in all cases we'll return:
boxes, objs, rels, pred_scores

"""

import numpy as np
import torch
from lib.pytorch_misc import unravel_index
from lib.fpn.box_utils import bbox_overlaps
# from ad3 import factor_graph as fg
from time import time
import pdb
from torch.autograd import Variable

# return filter_dets(bboxes, result.obj_scores,
#                    result.obj_preds, rel_inds[:, 1:], rel_rep, self.return_top100)

def filter_dets(boxes, obj_scores, obj_classes, rel_inds, pred_scores, rel_dists=None, ent_dists=None, return_top100=False, training=False):
# def filter_dets(boxes, obj_scores, obj_classes, rel_inds, pred_scores, return_top100=False, training=False):
    """
    Filters detections....
    :param boxes: [num_box, topk, 4] if bbox regression else [num_box, 4]
    :param obj_scores: [num_box] probabilities for the scores
    :param obj_classes: [num_box] class labels for the topk
    :param rel_inds: [num_rel, 2] TENSOR consisting of (im_ind0, im_ind1)
    :param pred_scores: [topk, topk, num_rel, num_predicates]
    :param use_nms: True if use NMS to filter dets.
    :return: boxes, objs, rels, pred_scores

    """
    if boxes.dim() != 2:
        raise ValueError("Boxes needs to be [num_box, 4] but its {}".format(boxes.size()))

    num_box = boxes.size(0)
    assert obj_scores.size(0) == num_box

    assert obj_classes.size() == obj_scores.size()
    num_rel = rel_inds.size(0)
    assert rel_inds.size(1) == 2
    assert pred_scores.size(0) == num_rel

    obj_scores0 = obj_scores.data[rel_inds[:,0]]
    obj_scores1 = obj_scores.data[rel_inds[:,1]]

    pred_scores_max, pred_classes_argmax = pred_scores.data[:,1:].max(1)
    pred_classes_argmax = pred_classes_argmax + 1

    rel_scores_argmaxed = pred_scores_max * obj_scores0 * obj_scores1
    rel_scores_vs, rel_scores_idx = torch.sort(rel_scores_argmaxed.view(-1), dim=0, descending=True)

    # obj_scores_np = obj_scores.data.cpu().numpy()
    # objs_np = obj_classes.data.cpu().numpy()
    # boxes_out = boxes.data.cpu().numpy()
    # try:
    rel_dists_sorted = rel_dists[rel_scores_idx]
    # except Exception as e:
    #     print(e)
    #     print("")
    split = 100

    if return_top100:
        # rels_b_100 = rel_inds[rel_scores_idx[:100]].cpu().numpy()
        # pred_scores_sorted_b_100 = pred_scores[rel_scores_idx[:100]].data.cpu().numpy()
        # rels_a_100 = rel_inds[rel_scores_idx[100:]].cpu().numpy()
        # pred_scores_sorted_a_100 = pred_scores[rel_scores_idx[100:]].data.cpu().numpy()

        rel_b_dists = rel_dists_sorted[:split]

        rels_b_100 = rel_inds[rel_scores_idx[:100]]
        pred_scores_sorted_b_100 = pred_scores[rel_scores_idx[:100]]
        rel_scores_idx_b_100 = rel_scores_idx[:100]

        if rel_scores_idx.size()[0] <= 100:
            rel_a_dists = torch.Tensor([]).long().cuda()
            rels_a_100 = torch.Tensor([]).long().cuda()
            pred_scores_sorted_a_100 = Variable(torch.Tensor([]).long().cuda())
            rel_scores_idx_a_100 = torch.Tensor([]).long().cuda()
        else:
            rels_a_100 = rel_inds[rel_scores_idx[100:]]
            pred_scores_sorted_a_100 = pred_scores[rel_scores_idx[100:]]
            rel_scores_idx_a_100 = rel_scores_idx[100:]
            rel_a_dists = rel_dists_sorted[split:]

        if training:
            return boxes, obj_classes, obj_scores, rels_b_100, pred_scores_sorted_b_100, rels_a_100, \
               pred_scores_sorted_a_100, rel_scores_idx_b_100, rel_scores_idx_a_100, rel_b_dists, rel_a_dists, ent_dists
        else:
            # return boxes.data.cpu().numpy(), obj_classes.data.cpu().numpy(), obj_scores.data.cpu().numpy(), \
            #        rels_b_100.cpu().numpy(), pred_scores_sorted_b_100.data.cpu().numpy(), rels_a_100.cpu().numpy(), \
            #    pred_scores_sorted_a_100.data.cpu().numpy(), rel_scores_idx_b_100, rel_scores_idx_a_100

           return boxes.data.cpu().numpy(), obj_classes.data.cpu().numpy(), obj_scores.data.cpu().numpy(), \
                   rels_b_100.cpu().numpy(), pred_scores_sorted_b_100.data.cpu().numpy(), rels_a_100.cpu().numpy(), \
               pred_scores_sorted_a_100.data.cpu().numpy(), rel_scores_idx_b_100.cpu().numpy(), rel_scores_idx_a_100.cpu().numpy(), \
                  rel_b_dists, rel_a_dists, ent_dists
    else:
        rels = rel_inds[rel_scores_idx]
        pred_scores_sorted = pred_scores[rel_scores_idx]

        if training:
            return boxes, obj_classes, obj_scores, rels, pred_scores_sorted, rel_dists_sorted, ent_dists
        else:
            return boxes.data.cpu().numpy(), obj_classes.data.cpu().numpy(), obj_scores.data.cpu().numpy(), rels.cpu().numpy(), pred_scores_sorted.data.cpu().numpy(), rel_dists_sorted, ent_dists

# def _get_similar_boxes(boxes, obj_classes_topk, nms_thresh=0.3):
#     """
#     Assuming bg is NOT A LABEL.
#     :param boxes: [num_box, topk, 4] if bbox regression else [num_box, 4]
#     :param obj_classes: [num_box, topk] class labels
#     :return: num_box, topk, num_box, topk array containing similarities.
#     """
#     topk = obj_classes_topk.size(1)
#     num_box = boxes.size(0)
#
#     box_flat = boxes.view(-1, 4) if boxes.dim() == 3 else boxes[:, None].expand(
#         num_box, topk, 4).contiguous().view(-1, 4)
#     jax = bbox_overlaps(box_flat, box_flat).data > nms_thresh
#     # Filter out things that are not gonna compete.
#     classes_eq = obj_classes_topk.data.view(-1)[:, None] == obj_classes_topk.data.view(-1)[None, :]
#     jax &= classes_eq
#     boxes_are_similar = jax.view(num_box, topk, num_box, topk)
#     return boxes_are_similar.cpu().numpy().astype(np.bool)
