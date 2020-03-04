"""
Let's get the relationships yo
"""

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torch.nn import functional as F
from lib.surgery_sgcls import filter_dets
from lib.fpn.proposal_assignments.rel_assignments import rel_assignments
from lib.pytorch_misc import arange
from lib.object_detector import filter_det
from lib.stanford_rel import RelModel
import numpy as np

MODES = ('sgdet', 'sgcls', 'predcls')

SIZE=512

class RelModelStanford(RelModel):
    """
    RELATIONSHIPS
    """

    def __init__(self, classes, rel_classes, mode='sgdet', num_gpus=1, require_overlap_det=True,
                 use_resnet=False, use_proposals=False, return_top100=False, **kwargs):
        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param num_gpus: how many GPUS 2 use
        """
        super(RelModelStanford, self).__init__(classes, rel_classes, mode=mode, num_gpus=num_gpus,
                                               require_overlap_det=require_overlap_det,
                                               use_resnet=use_resnet,
                                               nl_obj=0, nl_edge=0, use_proposals=use_proposals, thresh=0.01,
                                               pooling_dim=4096)

        del self.context
        del self.post_lstm
        del self.post_emb

        self.return_top100 = return_top100


        self.rel_fc = nn.Linear(SIZE, self.num_rels)
        self.obj_fc = nn.Linear(SIZE, self.num_classes)

        self.obj_unary = nn.Linear(self.obj_dim, SIZE)
        self.edge_unary = nn.Linear(4096, SIZE)


        self.edge_gru = nn.GRUCell(input_size=SIZE, hidden_size=SIZE)
        self.node_gru = nn.GRUCell(input_size=SIZE, hidden_size=SIZE)

        self.n_iter = 3

        self.sub_vert_w_fc = nn.Sequential(nn.Linear(SIZE*2, 1), nn.Sigmoid())
        self.obj_vert_w_fc = nn.Sequential(nn.Linear(SIZE*2, 1), nn.Sigmoid())
        self.out_edge_w_fc = nn.Sequential(nn.Linear(SIZE*2, 1), nn.Sigmoid())

        self.in_edge_w_fc = nn.Sequential(nn.Linear(SIZE*2, 1), nn.Sigmoid())

    def message_pass(self, rel_rep, obj_rep, rel_inds):
        """

        :param rel_rep: [num_rel, fc]
        :param obj_rep: [num_obj, fc]
        :param rel_inds: [num_rel, 2] of the valid relationships
        :return: object prediction [num_obj, 151], bbox_prediction [num_obj, 151*4]
                and rel prediction [num_rel, 51]
        """
        # [num_obj, num_rel] with binary!
        numer = torch.arange(0, rel_inds.size(0)).long().cuda(rel_inds.get_device())

        objs_to_outrels = rel_rep.data.new(obj_rep.size(0), rel_rep.size(0)).zero_()
        objs_to_outrels.view(-1)[rel_inds[:, 0] * rel_rep.size(0) + numer] = 1
        objs_to_outrels = Variable(objs_to_outrels)

        objs_to_inrels = rel_rep.data.new(obj_rep.size(0), rel_rep.size(0)).zero_()
        objs_to_inrels.view(-1)[rel_inds[:, 1] * rel_rep.size(0) + numer] = 1
        objs_to_inrels = Variable(objs_to_inrels)

        hx_rel = Variable(rel_rep.data.new(rel_rep.size(0), SIZE).zero_(), requires_grad=False)
        hx_obj = Variable(obj_rep.data.new(obj_rep.size(0), SIZE).zero_(), requires_grad=False)

        vert_factor = [self.node_gru(obj_rep, hx_obj)]
        edge_factor = [self.edge_gru(rel_rep, hx_rel)]

        for i in range(3):
            # compute edge context
            sub_vert = vert_factor[i][rel_inds[:, 0]]
            obj_vert = vert_factor[i][rel_inds[:, 1]]
            weighted_sub = self.sub_vert_w_fc(
                torch.cat((sub_vert, edge_factor[i]), 1)) * sub_vert
            weighted_obj = self.obj_vert_w_fc(
                torch.cat((obj_vert, edge_factor[i]), 1)) * obj_vert

            edge_factor.append(self.edge_gru(weighted_sub + weighted_obj, edge_factor[i]))

            # Compute vertex context
            pre_out = self.out_edge_w_fc(torch.cat((sub_vert, edge_factor[i]), 1)) * \
                      edge_factor[i]
            pre_in = self.in_edge_w_fc(torch.cat((obj_vert, edge_factor[i]), 1)) * edge_factor[
                i]

            vert_ctx = objs_to_outrels @ pre_out + objs_to_inrels @ pre_in
            vert_factor.append(self.node_gru(vert_ctx, vert_factor[i]))

        # woohoo! done
        return self.obj_fc(vert_factor[-1]), self.rel_fc(edge_factor[-1])
               # self.box_fc(vert_factor[-1]).view(-1, self.num_classes, 4), \
               # self.rel_fc(edge_factor[-1])

    def forward(self, x, im_sizes, image_offset,
                gt_boxes=None, gt_classes=None, gt_rels=None, proposals=None, train_anchor_inds=None,
                return_fmap=False):
        """
        Forward pass for detection
        :param x: Images@[batch_size, 3, IM_SIZE, IM_SIZE]
        :param im_sizes: A numpy array of (h, w, scale) for each image.
        :param image_offset: Offset onto what image we're on for MGPU training (if single GPU this is 0)
        :param gt_boxes:

        Training parameters:
        :param gt_boxes: [num_gt, 4] GT boxes over the batch.
        :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
        :param train_anchor_inds: a [num_train, 2] array of indices for the anchors that will
                                  be used to compute the training loss. Each (img_ind, fpn_idx)
        :return: If train:
            scores, boxdeltas, labels, boxes, boxtargets, rpnscores, rpnboxes, rellabels

            if test:
            prob dists, boxes, img inds, maxscores, classes

        """
        result = self.detector(x, im_sizes, image_offset, gt_boxes, gt_classes, gt_rels, proposals,
                               train_anchor_inds, return_fmap=True)

        if result.is_none():
            return ValueError("heck")

        im_inds = result.im_inds - image_offset
        boxes = result.rm_box_priors

        if self.training and result.rel_labels is None:
            assert self.mode == 'sgdet'
            result.rel_labels = rel_assignments(im_inds.data, boxes.data, result.rm_obj_labels.data,
                                                gt_boxes.data, gt_classes.data, gt_rels.data,
                                                image_offset, filter_non_overlap=True, num_sample_per_gt=1)
        rel_inds = self.get_rel_inds(result.rel_labels, im_inds, boxes)
        rois = torch.cat((im_inds[:, None].float(), boxes), 1)
        visual_rep = self.visual_rep(result.fmap, rois, rel_inds[:, 1:])

        result.obj_fmap = self.obj_feature_map(result.fmap.detach(), rois)

        # Now do the approximation WHEREVER THERES A VALID RELATIONSHIP.
        result.rm_obj_dists, result.rel_dists = self.message_pass(
            F.relu(self.edge_unary(visual_rep)), self.obj_unary(result.obj_fmap), rel_inds[:, 1:])

        # result.box_deltas_update = box_deltas

        result.rel_inds = rel_inds

        # if self.training:
        #     return result

        def index2class(rels_index, rels_classes):
            return_index = []
            for index in rels_index:
                index = index.data.cpu().numpy()[0]
                return_index.append(rels_classes[index])
            return torch.from_numpy(np.asarray(return_index))

        dict_gt = {}
        gt_sub_list = index2class(gt_rels[:, 1], gt_classes.data[:, 1])
        gt_obj_list = index2class(gt_rels[:, 2], gt_classes.data[:, 1])

        for i in range(gt_rels.size(0)):
            if (int(gt_sub_list[i]), int(gt_obj_list[i]), int(gt_rels[i, 3])) in dict_gt:
                dict_gt[(int(gt_sub_list[i]), int(gt_obj_list[i]), int(gt_rels[i, 1]), int(gt_rels[i, 2]), int(gt_rels[i, 3]))] += 1
            else:
                dict_gt[(int(gt_sub_list[i]), int(gt_obj_list[i]), int(gt_rels[i, 1]), int(gt_rels[i, 2]), int(gt_rels[i, 3]))] = 1

        if self.training:

            # For bug0 >>>>>>>>>
            if self.mode == "sgdet":
                probs = F.softmax(result.rm_obj_dists, 1)
                nms_mask = result.rm_obj_dists.data.clone()
                nms_mask.zero_()
                for c_i in range(1, result.rm_obj_dists.size(1)):
                    scores_ci = probs.data[:, c_i]
                    boxes_ci = result.boxes_all.data[:, c_i]

                    keep = apply_nms(scores_ci, boxes_ci,
                                     pre_nms_topn=scores_ci.size(0), post_nms_topn=scores_ci.size(0),
                                     nms_thresh=0.3)
                    nms_mask[:, c_i][keep] = 1

                result.obj_preds = Variable(nms_mask * probs.data, volatile=True)[:, 1:].max(1)[1] + 1
            else:
                result.obj_preds = result.rm_obj_dists[:,1:].max(1)[1] + 1
            # For bug0 <<<<<<<<

            if self.return_top100:
                if self.mode == 'predcls':
                    # Hack to get the GT object labels
                    result.obj_scores = result.rm_obj_dists.data.new(gt_classes.size(0)).fill_(1)
                    result.obj_preds = gt_classes.data[:, 1]
                elif self.mode == 'sgdet':
                    order, obj_scores, obj_preds = filter_det(F.softmax(result.rm_obj_dists),
                                                              result.boxes_all,
                                                              start_ind=0,
                                                              max_per_img=100,
                                                              thresh=0.00,
                                                              pre_nms_topn=6000,
                                                              post_nms_topn=300,
                                                              nms_thresh=0.3,
                                                              nms_filter_duplicates=True)
                    idx, perm = torch.sort(order)
                    result.obj_preds = rel_inds.new(result.rm_obj_dists.size(0)).fill_(1)
                    result.obj_scores = result.rm_obj_dists.data.new(result.rm_obj_dists.size(0)).fill_(0)
                    result.obj_scores[idx] = obj_scores.data[perm]
                    result.obj_preds[idx] = obj_preds.data[perm]
                else:
                    scores_nz = F.softmax(result.rm_obj_dists).data
                    scores_nz[:, 0] = 0.0
                    result.obj_scores, score_ord = scores_nz[:, 1:].sort(dim=1, descending=True)
                    result.obj_preds = score_ord[:, 0] + 1
                    result.obj_scores = result.obj_scores[:, 0]

                result.obj_preds = Variable(result.obj_preds)
                result.obj_scores = Variable(result.obj_scores)

                # Set result's bounding boxes to be size
                # [num_boxes, topk, 4] instead of considering every single object assignment.
                twod_inds = arange(result.obj_preds.data) * self.num_classes + result.obj_preds.data

                if self.mode == 'sgdet':
                    bboxes = result.boxes_all.view(-1, 4)[twod_inds].view(result.boxes_all.size(0), 4)
                else:
                    # Boxes will get fixed by filter_dets function.
                    bboxes = result.rm_box_priors
                rel_rep = F.softmax(result.rel_dists)

                if rel_inds[:, 0].max() - rel_inds[:, 0].min() + 1 == 1:
                    # return result, filter_dets(bboxes, result.obj_scores,
                    #                            result.obj_preds, rel_inds[:, 1:], rel_rep, self.return_top100,
                    #                            self.training)
                    return result, filter_dets(bboxes, result.obj_scores,
                                               result.obj_preds, rel_inds[:, 1:], rel_rep, rel_dists=result.rel_dists,
                                               ent_dists=result.rm_obj_dists,
                                               return_top100=self.return_top100, training=self.training), dict_gt

                # -----------------------------------Above: 1 batch_size, Below: Multiple batch_size------------------
                #  assume rel_inds[:, 0] is from 0 to num_img-1
                num_img = rel_inds[:, 0].max() - rel_inds[:, 0].min() + 1
                rel_ind_per_img = []
                obj_ind_per_img = []
                for i in range(num_img):
                    rel_ind_cur_img = torch.nonzero(rel_inds[:, 0] == i).squeeze(-1)
                    rel_ind_per_img.append(rel_ind_cur_img)
                    obj_ind_cur_img = rel_inds[rel_ind_cur_img][:, 1]
                    obj_ind_cur_img_norepeat = []
                    for j in range(obj_ind_cur_img.size(0)):
                        obj_ind_cur_img_norepeat += [obj_ind_cur_img[j]] if obj_ind_cur_img[
                                                                                j] not in obj_ind_cur_img_norepeat else []
                    obj_ind_per_img.append(torch.Tensor(obj_ind_cur_img_norepeat).type_as(rel_ind_cur_img))

                rels_b_100_all = []
                pred_scores_sorted_b_100_all = []
                rels_a_100_all = []
                pred_scores_sorted_a_100_all = []
                rel_scores_idx_b_100_all = []
                rel_scores_idx_a_100_all = []
                rel_dists_b_all = []
                rel_dists_a_all = []

                for i in range(len(rel_ind_per_img)):
                    # boxes, obj_classes, obj_scores, rels_b_100, pred_scores_sorted_b_100, rels_a_100, \
                    # pred_scores_sorted_a_100, rel_scores_idx_b_100, rel_scores_idx_a_100 = filter_dets(bboxes,
                    #                                                                                    result.obj_scores,
                    #                                                                                    result.obj_preds,
                    #                                                                                    rel_inds[
                    #                                                                                        rel_ind_per_img[
                    #                                                                                            i]][:,
                    #                                                                                    1:], rel_rep[
                    #                                                                                             rel_ind_per_img[
                    #                                                                                                 i]][
                    #                                                                                         :, 1:],
                    #                                                                                    self.return_top100,
                    #                                                                                    self.training)

                    # boxes, obj_classes, obj_scores, rels_b_100, pred_scores_sorted_b_100, rels_a_100, \
                    # pred_scores_sorted_a_100, rel_scores_idx_b_100, rel_scores_idx_a_100 = filter_dets(bboxes,
                    # result.obj_scores, result.obj_preds, rel_inds[rel_ind_per_img[i]][:,1:], rel_rep[rel_ind_per_img[i]],
                    #                                                                                    self.return_top100,
                    #                                                                                    self.training)
                    (boxes, obj_classes, obj_scores, rels_b_100, pred_scores_sorted_b_100, rels_a_100, \
                     pred_scores_sorted_a_100, rel_scores_idx_b_100, rel_scores_idx_a_100, rel_dists_b, rel_dists_a,
                     ent_dists) = filter_dets(bboxes, result.obj_scores,
                                              result.obj_preds, rel_inds[rel_ind_per_img[i]][:, 1:],
                                              rel_rep[rel_ind_per_img[i]],
                                              rel_dists=result.rel_dists,
                                              ent_dists=result.rm_obj_dists,
                                              return_top100=self.return_top100,
                                              training=self.training)

                    # pdb.set_trace()

                    rels_b_100_all.append(
                        torch.cat((i * torch.ones(rels_b_100.size(0), 1).type_as(rels_b_100), rels_b_100), dim=1))
                    pred_scores_sorted_b_100_all.append(pred_scores_sorted_b_100)
                    rel_dists_b_all.append(rel_dists_b)

                    try:
                        rels_a_100_all.append(
                            torch.cat((i * torch.ones(rels_a_100.size(0), 1).type_as(rels_a_100), rels_a_100), dim=1))
                    except:
                        rels_a_100_all.append(torch.Tensor([]).long().cuda())

                    pred_scores_sorted_a_100_all.append(pred_scores_sorted_a_100)
                    rel_dists_a_all.append(rel_dists_a)
                    rel_scores_idx_b_100_all.append(rel_ind_per_img[i][rel_scores_idx_b_100])
                    # pdb.set_trace()
                    try:
                        rel_scores_idx_a_100_all.append(rel_ind_per_img[i][rel_scores_idx_a_100])
                    except:
                        rel_scores_idx_a_100_all.append(torch.Tensor([]).long().cuda())

                # pdb.set_trace()
                rels_b_100_all = torch.cat(rels_b_100_all, dim=0)
                pred_scores_sorted_b_100_all = torch.cat(pred_scores_sorted_b_100_all, dim=0)
                rel_scores_idx_b_100_all = torch.cat(rel_scores_idx_b_100_all, 0)
                rel_dists_b_all = torch.cat(rel_dists_b_all, dim=0)

                try:
                    rels_a_100_all = torch.cat(rels_a_100_all, 0)
                except:
                    rels_a_100_all = torch.Tensor([]).long().cuda()

                try:
                    pred_scores_sorted_a_100_all = torch.cat(pred_scores_sorted_a_100_all, 0)
                except:
                    pred_scores_sorted_a_100_all = Variable(torch.Tensor([]).long().cuda())

                try:
                    rel_scores_idx_a_100_all = torch.cat(rel_scores_idx_a_100_all, 0)
                except:
                    rel_scores_idx_a_100_all = torch.Tensor([]).long().cuda()

                try:
                    rel_dists_a_all = torch.cat(rel_dists_a_all, dim=0)
                except:
                    rel_dists_a_all = torch.Tensor([]).long().cuda()

                return result, [boxes, obj_classes, obj_scores, rels_b_100_all, pred_scores_sorted_b_100_all,
                                rels_a_100_all,
                                pred_scores_sorted_a_100_all, rel_scores_idx_b_100_all, rel_scores_idx_a_100_all,
                                rel_dists_b_all, rel_dists_a_all, ent_dists], dict_gt

            else:
                # return result, []
                return result


        # Decode here ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.mode == 'predcls':
            # Hack to get the GT object labels
            result.obj_scores = result.rm_obj_dists.data.new(gt_classes.size(0)).fill_(1)
            result.obj_preds = gt_classes.data[:, 1]
        elif self.mode == 'sgdet':
            order, obj_scores, obj_preds= filter_det(F.softmax(result.rm_obj_dists),
                                                              result.boxes_all,
                                                              start_ind=0,
                                                              max_per_img=100,
                                                              thresh=0.00,
                                                              pre_nms_topn=6000,
                                                              post_nms_topn=300,
                                                              nms_thresh=0.3,
                                                              nms_filter_duplicates=True)
            idx, perm = torch.sort(order)
            result.obj_preds = rel_inds.new(result.rm_obj_dists.size(0)).fill_(1)
            result.obj_scores = result.rm_obj_dists.data.new(result.rm_obj_dists.size(0)).fill_(0)
            result.obj_scores[idx] = obj_scores.data[perm]
            result.obj_preds[idx] = obj_preds.data[perm]
        else:
            scores_nz = F.softmax(result.rm_obj_dists).data
            scores_nz[:, 0] = 0.0
            result.obj_scores, score_ord = scores_nz[:, 1:].sort(dim=1, descending=True)
            result.obj_preds = score_ord[:,0] + 1
            result.obj_scores = result.obj_scores[:,0]

        result.obj_preds = Variable(result.obj_preds)
        result.obj_scores = Variable(result.obj_scores)

        # Set result's bounding boxes to be size
        # [num_boxes, topk, 4] instead of considering every single object assignment.
        twod_inds = arange(result.obj_preds.data) * self.num_classes + result.obj_preds.data

        if self.mode == 'sgdet':
            bboxes = result.boxes_all.view(-1, 4)[twod_inds].view(result.boxes_all.size(0), 4)
        else:
            # Boxes will get fixed by filter_dets function.
            bboxes = result.rm_box_priors
        rel_rep = F.softmax(result.rel_dists)

        # return filter_dets(bboxes, result.obj_scores,
        #                    result.obj_preds, rel_inds[:, 1:], rel_rep, self.return_top100)

        # dict_gt = {}
        # for i in range(gt_rels.size(0)):
        #     if (int(gt_rels[i, 1]), int(gt_rels[i, 2])) in dict_gt:
        #         dict_gt[(int(gt_rels[i, 1]), int(gt_rels[i, 2]))].append(int(gt_rels[i, 3]))
        #     else:
        #         dict_gt[(int(gt_rels[i, 1]), int(gt_rels[i, 2]))] = [int(gt_rels[i, 3])]

        # return dict_gt, filter_dets(bboxes, result.obj_scores,
        #                             result.obj_preds, rel_inds[:, 1:], rel_rep, self.return_top100)

        return dict_gt, filter_dets(bboxes, result.obj_scores, result.obj_preds, rel_inds[:, 1:],
                            rel_rep, rel_dists=result.rel_dists, ent_dists=result.rm_obj_dists,
                            return_top100=self.return_top100, training=False)
