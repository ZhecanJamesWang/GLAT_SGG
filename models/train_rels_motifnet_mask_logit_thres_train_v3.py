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
# from lib.stanford_model import RelModelStanford as RelModel
from lib.motifnet_model import RelModel
# from lib.kern_model import KERN

from lib.glat import GLATNET
import pdb
from torch.autograd import Variable
import copy
from scipy.special import softmax
import torch.optim.lr_scheduler as lr_scheduler

#--------updated--------
import sys
import os
import math


codebase = '../../'
sys.path.append(codebase)
exp_name = 'motif'


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# conf = ModelConfig()
#--------updated--------
conf = ModelConfig()

# We use tensorboard to observe results and decrease learning rate manually. If you want to use TB, you need to install TensorFlow fist.
if conf.tb_log_dir is not None:
    from tensorboardX import SummaryWriter
    if not os.path.exists(conf.tb_log_dir):
        os.makedirs(conf.tb_log_dir)
    writer = SummaryWriter(log_dir=conf.tb_log_dir)
    use_tb = True
else:
    use_tb = False

train, val, _ = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                          use_proposals=conf.use_proposals,
                          filter_non_overlap=conf.mode == 'sgdet')

ind_to_predicates = train.ind_to_predicates # ind_to_predicates[0] means no relationship
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
                    limit_vision=limit_vision,
                    return_top100=True,
                    return_unbias_logit=True
                    )


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
#                     return_top100=True,
#                     return_unbias_logit=True
#                     )


model = GLATNET(vocab_num=[52, 153],
                feat_dim=300,
                nhid_glat_g=300,
                nhid_glat_l=300,
                nout=300,
                dropout=0.1,
                nheads=8,
                blank=152,
                types=[2]*6)


# Freeze all the motif model
for n, param in detector.named_parameters():
    param.requires_grad = False

# Freeze the detector
# for n, param in detector.detector.named_parameters():
#     param.requires_grad = False

print(print_para(detector), flush=True)


def get_optim(lr, last_epoch=-1):
    # Lower the learning rate on the VGG fully connected layers by 1/10th. It's a hack, but it helps
    # stabilize the models.
    # fc_params = [p for n,p in detector.named_parameters() if n.startswith('roi_fmap') and p.requires_grad]
    # non_fc_params = [p for n,p in detector.named_parameters() if not n.startswith('roi_fmap') and p.requires_grad]
    # params = [{'params': fc_params, 'lr': lr / 10.0}, {'params': non_fc_params}]
    params = model.parameters()
    if conf.adam:
        optimizer = optim.Adam(params, weight_decay=conf.adamwd, lr=lr, eps=1e-3)
    else:
        optimizer = optim.SGD(params, weight_decay=conf.l2, lr=lr, momentum=0.9)

    # scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1,
    #                               verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3, last_epoch=last_epoch)

    return optimizer, scheduler

ckpt = torch.load(conf.ckpt)
print("Loading EVERYTHING from motifnet", conf.ckpt)
optimistic_restore(detector, ckpt['state_dict'])
detector.cuda()
start_epoch = -1

# if conf.ckpt.split('-')[-2].split('/')[-1] == 'vgrel':
#     print("Loading EVERYTHING")
#     start_epoch = ckpt['epoch']
#
#     if not optimistic_restore(detector, ckpt['state_dict']):
#         start_epoch = -1
#         # optimistic_restore(detector.detector, torch.load('checkpoints/vgdet/vg-28.tar')['state_dict'])
# else:
#     start_epoch = -1
#     optimistic_restore(detector.detector, ckpt['state_dict'])
#
#     detector.roi_fmap[1][0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
#     detector.roi_fmap[1][3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
#     detector.roi_fmap[1][0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
#     detector.roi_fmap[1][3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])
#
#     detector.roi_fmap_obj[0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
#     detector.roi_fmap_obj[3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
#     detector.roi_fmap_obj[0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
#     detector.roi_fmap_obj[3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])



print('finish pretrained GLAT loading')
# # model.load_state_dict(ckpt_glat['model'])
if conf.resume_training:
    ckpt_glat = torch.load(conf.resume_ckpt)
    optimistic_restore(model, ckpt_glat['state_dict'])
    model.cuda()
    start_epoch = ckpt_glat['epoch']
    optimizer, scheduler = get_optim(conf.lr, last_epoch=start_epoch)

else:
    # # ckpt_glat = torch.load('/home/haoxuan/code/GBERT/models/2019-10-31-03-13_2_2_2_2_2_2_concat_no_init_mask/best_test_node_mask_predicate_acc')
    # ---------------pretrained model mask ration 0.5
    # ckpt_glat = torch.load('/home/tangtangwzc/Common_sense/models/2019-11-03-17-51_2_2_2_2_2_2_concat_no_init_mask/best_test_node_mask_predicate_acc.pth')

    # ---------------pretrained model mask ration 0.3
    ckpt_glat = torch.load(
        '/home/tangtangwzc/Common_sense/models/2019-12-18-16-08_2_2_2_2_2_2_concat_no_init_mask/best_test_node_mask_predicate_acc.pth')

    # ckpt_glat = torch.load(
    #     '/home/tangtangwzc/Common_sense/models/2019-11-03-17-28_2_2_2_2_2_2_concat_no_init_mask/best_test_node_mask_predicate_acc.pth')

    # ---------------pretrained model mask ration 0.7
    # ckpt_glat = torch.load('/home/tangtangwzc/Common_sense/models/2019-11-07-23-50_2_2_2_2_2_2_concat_no_init_mask/best_test_node_mask_predicate_acc.pth')

    optimistic_restore(model, ckpt_glat['model'])
    model.cuda()
    start_epoch = -1
    optimizer, scheduler = get_optim(conf.lr, last_epoch=start_epoch)


def train_epoch(epoch_num, train_results, train_bias_logits, train_det_ress):
    detector.train()
    for n, param in detector.named_parameters():
        param.requires_grad = False
    model.train()
    tr = []
    start = time.time()
    accs = [0, 0]

    for b, batch in enumerate(train_loader):
        tr.append(train_batch(batch, train_results, train_bias_logits, train_det_ress, epoch_num=epoch_num, batch_num=b, accs=accs,
                              verbose=b % (conf.print_interval*10) == 0)) #b == 0))

        if b % conf.print_interval == 0 and b >= conf.print_interval:
            mn = pd.concat(tr[-conf.print_interval:], axis=1).mean(1)
            time_per_batch = (time.time() - start) / conf.print_interval
            print("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
                epoch_num, b, len(train_loader), time_per_batch, len(train_loader) * time_per_batch / 60))
            print(mn)
            print('acc of mask:', accs[0]*1.0 / accs[1])
            print('num of mask:', accs[1])
            print('-----------', flush=True)
            start = time.time()

            # break

    return pd.concat(tr, axis=1)


def transfer_result_cpu(result):
    from lib.object_detector import Result
    result_cpu = Result()
    result_cpu.obj_preds = result.obj_preds.cpu()
    result_cpu.rel_inds = result.rel_inds.cpu()

    result_cpu.rel_dists = result.rel_dists.cpu()
    result_cpu.rel_labels = result.rel_labels.cpu()

    result_cpu.rm_obj_dists = result.rm_obj_dists.cpu()
    result_cpu.rm_obj_labels = result.rm_obj_labels.cpu()
    return result_cpu

def transfer_logit_cpu(bias_logit):
    return bias_logit.cpu()

def transfer_det_cpu(det_res):
    det_res_cpu = []
    for i in det_res:
        det_res_cpu.append(i.cpu())
    return det_res_cpu


def transfer_result_gpu(result):
    from lib.object_detector import Result
    result_gpu = Result()
    result_gpu.obj_preds = result.obj_preds.cuda()
    result_gpu.rel_inds = result.rel_inds.cuda()

    result_gpu.rel_dists = result.rel_dists.cuda()
    result_gpu.rel_labels = result.rel_labels.cuda()

    result_gpu.rm_obj_dists = result.rm_obj_dists.cuda()
    result_gpu.rm_obj_labels = result.rm_obj_labels.cuda()
    return result_gpu

def transfer_logit_gpu(bias_logit):
    return bias_logit.cuda()

def transfer_det_gpu(det_res):
    det_res_gpu = []
    for i in det_res:
        det_res_gpu.append(i.cuda())
    return det_res_gpu

def train_batch(b, train_results, train_bias_logits, train_det_ress, epoch_num, batch_num, accs, verbose=False):
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
    if epoch_num == 0:
        result, bias_logit, det_res = detector[b]
        train_results.append(transfer_result_cpu(result))
        train_bias_logits.append(transfer_logit_cpu(bias_logit))
        train_det_ress.append(transfer_det_cpu(det_res))
    else:
        result = transfer_result_gpu(train_results[batch_num])
        bias_logit = transfer_logit_gpu(train_bias_logits[batch_num])
        det_res = transfer_det_gpu(train_det_ress[batch_num])

    # pdb.set_trace()
    # result.rm_obj_dists(num_entities, 151)  result.obj_preds(num_entities)  result.rm_obj_labels(num_entities)
    # result.rel_dists(num_predicates, 51)  result.rel_labels(num_predicates)
    # result.rel_inds(num_predicates, 3) 3
    # pdb.set_trace()
    # rel_inds obj_idx?-> global idx

    # t1 = time.time()
    # print('base model time', t1-t0)

    if conf.return_top100 and len(det_res) != 0:

        pred_entry = {
            'pred_classes': result.obj_preds,  # (num_entities) Tensor Variable
            'pred_rel_inds': det_res[3],  # (num_predicates, 3) Tensor Variable
            'rel_scores': det_res[4],  # (num_predicates, 51) Tensor Variable
        }
    else:
        pred_entry = {
            'pred_classes': result.obj_preds,   # (num_entities) Tensor Variable
            'pred_rel_inds': result.rel_inds,  # (num_predicates, 3) Tensor Variable
            'rel_scores': result.rel_dists,   # (num_predicates, 51) Tensor Variable
        }

    threshold = 0.35
    b_100_idx = det_res[-2]
    bias_logit_norm = F.softmax(bias_logit[:, 1:], dim=1)
    bias_logit_norm_inalltop = bias_logit_norm[b_100_idx]

    num_predicates = []
    idx_predicates_inalltop = []
    for i in range(conf.batch_size):
        idx_predicates_inalltop.append(torch.nonzero(det_res[3][:, 0] == i)[:, 0].tolist())
        num_predicate = int((result.rel_inds[:, 0] == i).sum())
        if num_predicate >= 100:
            num_predicates.append(100)
        else:
            num_predicates.append(num_predicate)
        assert num_predicates[i] == int((det_res[3][:, 0] == i).sum())

    keep_nums = []
    for i in range(len(num_predicates)):
        keep_num = math.ceil(num_predicates[i]*0.3) if num_predicates[i] <= 100 else 30
        keep_nums.append(keep_num)

    mask_idxs_inalltop = []
    input_pred_idxs_inalltop = []

    for i in range(len(num_predicates)):
        keep_idx_inalltop = list(range(keep_nums[i]))
        keep_idx_inalltop = [keep_idx+sum(num_predicates[:i]) for keep_idx in keep_idx_inalltop] if i>0 else keep_idx_inalltop

        idx_predicates_inalltop_i = idx_predicates_inalltop[i]
        mask_idx_initop = torch.nonzero(bias_logit_norm_inalltop[idx_predicates_inalltop_i, :].max(1)[0] < threshold)
        if len(mask_idx_initop) != 0:
            mask_idx_initop = mask_idx_initop[:, 0].data
            mask_idx_inalltop = mask_idx_initop + sum(num_predicates[:i]) if i > 0 else mask_idx_initop
            mask_idx_inalltop = mask_idx_inalltop.tolist()
            mask_idxs_inalltop += mask_idx_inalltop
            mask_in_keep_num = len(torch.nonzero(mask_idx_initop < keep_nums[i]))
            input_pred_num = len(mask_idx_initop) + keep_nums[i] - mask_in_keep_num
        else:
            mask_idx_inalltop = []
            mask_in_keep_num = 0
            input_pred_num = keep_nums[i]


        for pred_id in range(input_pred_num):
            if pred_id < keep_nums[i]:
                input_pred_idxs_inalltop.append(keep_idx_inalltop[pred_id])
            else:
                input_pred_idxs_inalltop.append(mask_idx_inalltop[pred_id-keep_nums[i]+mask_in_keep_num])

    pred_entry['pred_rel_inds'] = pred_entry['pred_rel_inds'][input_pred_idxs_inalltop]
    pred_entry['rel_scores'] = pred_entry['rel_scores'][input_pred_idxs_inalltop]

    mask_idxs_ininput = torch.nonzero(bias_logit_norm_inalltop[input_pred_idxs_inalltop, :].max(1)[0] < threshold)[:, 0].data
    # mask_idxs_ininput = mask_idxs_ininput.tolist()

    # pdb.set_trace()

    assert len(mask_idxs_inalltop) == mask_idxs_ininput.size(0)

    # assert sum(num_predicates) == b_100_idx.size(0)
    # mask_idx = []
    # for b_idx, num_predicate in enumerate(num_predicates):
    #     start = sum(num_predicates[:b_idx])
    #     start = int(start + (1-0.3)*num_predicate)
    #     end = sum(num_predicates[:b_idx]) + num_predicate
    #     mask_idx.append(torch.Tensor(range(start, end)).long().cuda())
    # mask_idx = torch.cat(mask_idx)

    pred_entry = glat_postprocess(pred_entry, if_predicting=False, mask_idx=mask_idxs_ininput)

    rels_b_100 = pred_entry['pred_rel_inds']
    pred_scores_sorted_b_100 = pred_entry['rel_scores'][:, :-1]

    # b_100_idx abs coor

    result.obj_preds = pred_entry['pred_classes']

    # t2 = time.time()
    # print('glat model time', t2-t1)

    #
    # rels_total = sorted(set(zip(totla_idx, rels_total)))
    # rels_total = [rels[1] for rels in rels_total]
    #
    # pred_scores_sorted_total = sorted(set(zip(totla_idx, pred_scores_sorted_total)))
    # pred_scores_sorted_total = [pred_scores[1] for pred_scores in pred_scores_sorted_total]

    # merge two lists

    # if conf.return_top100:
    #     result.rel_inds = rels_total
    #     result.rel_dists = pred_scores_sorted_total
    # else:
    #     result.rel_inds = pred_entry['pred_rel_inds']
    #     result.rel_dists = pred_entry['rel_scores']

    losses = {}
    if conf.mode == "sgcls" or conf.mode == "sgdet":
    # if conf.use_ggnn_obj: # if not use ggnn obj, we just use scores of faster rcnn as their scores, there is no need to train
        losses['class_loss'] = F.cross_entropy(result.rm_obj_dists, result.rm_obj_labels)
    # losses['rel_loss'] = F.cross_entropy(result.rel_dists, result.rel_labels[:, -1])
    # pdb.set_trace()
    # losses['rel_loss'] = F.cross_entropy(pred_scores_sorted_b_100[mask_idxs_ininput], result.rel_labels[b_100_idx[mask_idxs_inalltop]][:, -1])
    losses['rel_loss'] = F.cross_entropy(pred_scores_sorted_b_100, result.rel_labels[b_100_idx[input_pred_idxs_inalltop]][:, -1])
    loss = sum(losses.values())

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm(
        [(n, p) for n, p in detector.named_parameters() if p.grad is not None],
        max_norm=conf.clip, verbose=verbose, clip=True)
    losses['total'] = loss
    optimizer.step()

    # pdb.set_trace()

    if mask_idxs_inalltop is not None:
        abs_mask_idx = b_100_idx[mask_idxs_inalltop]
        gt = result.rel_labels.data[abs_mask_idx][:, -1]
        pred = pred_scores_sorted_b_100[mask_idxs_ininput][:, 1:].max(1)[1].data + 1
        # pred = pred.type_as(gt)

        correct = int((pred == gt).sum())
        total = int(pred.size(0))
        accs[0] = accs[0] + correct
        accs[1] = accs[1] + total

    res = pd.Series({x: y.data[0] for x, y in losses.items()})
    return res


def val_epoch(epoch, eval_results, eval_logits, eval_det_ress):

    detector.eval()
    model.eval()
    evaluator_list = [] # for calculating recall of each relationship except no relationship
    evaluator_multiple_preds_list = []
    accs = [0, 0]

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
        val_batch(val_b, batch, evaluator, evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list,
                  epoch, eval_results, eval_logits, eval_det_ress, accs)

        # if val_b == 100:
        #     break


    recall = evaluator[conf.mode].print_stats()
    recall_mp = evaluator_multiple_preds[conf.mode].print_stats()

    mean_recall = calculate_mR_from_evaluator_list(evaluator_list, conf.mode)
    mean_recall_mp = calculate_mR_from_evaluator_list(evaluator_multiple_preds_list, conf.mode, multiple_preds=True)
    print('test acc of mask:', accs[0] * 1.0 / accs[1])
    writer.add_scalar('test acc of mask', accs[0] * 1.0 / accs[1], epoch)

    return recall, recall_mp, mean_recall, mean_recall_mp


def my_collate(total_data):
    blank_idx = 152
    max_length = 0
    sample_num = len(total_data['node_class'])
    for i in range(sample_num):
        max_length = max(max_length, total_data['node_class'][i].size(0))

    input_classes = []
    adjs = []
    node_types = []
    for i in range(sample_num):
        input_class = total_data['node_class'][i]
        adj = total_data['adj'][i]
        node_type = total_data['node_type'][i]
        pad_input_class = tensor2variable(blank_idx * torch.ones(max_length - input_class.size(0)).long().cuda())
        input_classes.append(torch.cat((input_class, pad_input_class), 0).unsqueeze(0))
        # input_classes.append(torch.cat((input_class, blank_idx*torch.ones(max_length-input_class.size(0)).long().cuda()), 0).unsqueeze(0))
        new_adj = torch.cat((adj, torch.zeros(max_length-adj.size(0), adj.size(1)).long().cuda()), 0)
        if max_length != new_adj.size(1):
            new_adj = torch.cat((new_adj, torch.zeros(new_adj.size(0), max_length-new_adj.size(1)).long().cuda()), 1)
        adjs.append(new_adj.unsqueeze(0))
        # pdb.set_trace()
        node_types.append(torch.cat((node_type, 2 * torch.ones(max_length-node_type.size(0)).long().cuda()), 0).unsqueeze(0))

    input_classes = torch.cat(input_classes, 0)
    adjs = torch.cat(adjs, 0)
    adjs_lbl = adjs
    adjs_con = torch.clamp(adjs, 0, 1)
    node_types = torch.cat(node_types, 0)

    return input_classes, adjs_con, adjs_lbl, node_types


def glat_wrapper(total_data):
    # Batch size assumed to be 1
    input_class, adjs_con, adjs_lbl, node_type = my_collate(total_data)
    if torch.is_tensor(input_class):
        input_class = Variable(input_class)
    if not torch.is_tensor(node_type):
        node_type = node_type.data
    if torch.is_tensor(adjs_con):
        adj_con = Variable(adjs_con)

    pred_label, pred_connect = model(input_class, adj_con, node_type)

    # pred_label_predicate = input_class[node_type == 0]
    # pred_label_entities = input_class[node_type == 1]

    # if input_class.size(0) ==1:
    #     pred_label_predicate = pred_label[0]  # flatten predicate (B*N, 51)
    #     pred_label_entities = pred_label[1]  # flatten entities
    # else:
    #     pred_label_predicate = []
    #     pred_label_entities = []
    #     predicate_num_list = [torch.nonzero(node_type[i] == 0).size(0) for i in range(node_type.size(0))]
    #     for i in range(len(predicate_num_list)):
    #         predicate_num_list[i] = predicate_num_list[i] + predicate_num_list[i-1] if i != 0 else predicate_num_list[i]
    #     for i in range(len(predicate_num_list)):
    #         pred_label_predicate.append()

    pred_label_predicate = pred_label[0]  # flatten predicate (B*N, 51)
    pred_label_entities = pred_label[1]  # flatten entities


    return pred_label_predicate, pred_label_entities
    # return pred_label_predicate.data.cpu().numpy(), pred_label_entities.data.cpu().numpy()


def numpy2cuda_dict(dict):
    for key, value in dict.items():
        dict[key] = torch.from_numpy(value).cuda()
    return dict

def cuda2numpy_dict(dict):
    for key, value in dict.items():
        # pdb.set_trace()
        if torch.is_tensor(value):
            dict[key] = value.cpu().numpy()
        else:
            dict[key] = value.data.cpu().numpy()
    return dict

def tensor2variable(input):
    if torch.is_tensor(input):
        input = Variable(input)
    return input

def variable2tensor(input):
    if not torch.is_tensor(input):
        input = input.data
    return input


def glat_postprocess(pred_entry, mask_idx, if_predicting=False):
    # pred_entry = {
    #     'pred_boxes': boxes_i * BOX_SCALE / IM_SCALE,  # (23, 4) (16, 4)
    #     'pred_classes': objs_i,  # (23,) (16,)
    #     'pred_rel_inds': rels_i,  # (506, 2) (240, 2)
    #     'obj_scores': obj_scores_i,  # (23,) (16,)
    #     'rel_scores': pred_scores_i,  # hack for now. (506, 51) (240, 51)
    # }

    if if_predicting:
        pred_entry = numpy2cuda_dict(pred_entry)

    pred_entry['rel_scores'] = tensor2variable(pred_entry['rel_scores'])
    pred_entry['pred_classes'] = tensor2variable(pred_entry['pred_classes'])

    pred_entry['rel_classes'] = torch.max(pred_entry['rel_scores'][:, 1:], dim=1)[1].unsqueeze(1) + 1
    if mask_idx is not None:
        pred_entry['rel_classes'][mask_idx] = 51
    pred_entry['rel_classes'] = variable2tensor(pred_entry['rel_classes'])
    # pdb.set_trace()
    pred_entry['pred_relations'] = torch.cat((pred_entry['pred_rel_inds'], pred_entry['rel_classes']), dim=1)

    # For SGCLS
    if conf.mode == "sgcls" or conf.mode == "sgdet":
        total_data, useless_entity_id = build_graph_structure(pred_entry, ind_to_classes, ind_to_predicates, if_predicting=if_predicting,
                                           sgclsdet=True)
    else:
        total_data = build_graph_structure(pred_entry, ind_to_classes, ind_to_predicates, if_predicting=if_predicting)

    pred_label_predicate, pred_label_entities = glat_wrapper(total_data)

    pred_entry['rel_scores'] = pred_label_predicate

    # For SGCLS
    # pred_entry['entity_scores'] = pred_label_entities


    if conf.mode == "sgcls" or conf.mode == "sgdet":
        pred_entry['pred_classes'] = pred_label_entities.max(1)[1]
        pred_entry['obj_scores_rm'] = pred_label_entities
        pred_entry['obj_scores'] = F.softmax(pred_label_entities, dim=1).max(1)[0]
        return pred_entry, useless_entity_id
    else:
        return pred_entry

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


def val_batch(batch_num, b, evaluator, evaluator_multiple_preds, evaluator_list,
              evaluator_multiple_preds_list, epoch_num, eval_results, eval_logits, eval_det_ress, accs):
    # det_res = detector[b]
    # dict_gt, det_res = detector[b]

    if epoch_num == 0:
        dict_gt, bias_logit, det_res = detector[b]
        eval_results.append(dict_gt)
        eval_logits.append(bias_logit)
        eval_det_ress.append(det_res)
    else:
        dict_gt = eval_results[batch_num]
        bias_logit = eval_logits[batch_num]
        det_res = eval_det_ress[batch_num]


    # if conf.num_gpus == 1:
    det_res = [det_res]

    for i, det in enumerate(det_res):

        if len(det) == 5 and not conf.return_top100:
            (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) = det
        else:
            (boxes_i, objs_i, obj_scores_i, rels_i_b100, pred_scores_i_b100, rels_i_a100, pred_scores_i_a100,
             rel_scores_idx_b100, rel_scores_idx_a100) = det

        gt_entry = {
            'gt_classes': val.gt_classes[batch_num + i].copy(), #(23,) (16,)
            'gt_relations': val.relationships[batch_num + i].copy(), #(29, 3) (6, 3)
            'gt_boxes': val.gt_boxes[batch_num + i].copy(), #(23, 4) (16, 4)
        }

        # val.relationships[batch_num + i]
        # np.argmax(pred_scores_i_b100[:, 1:], axis=1)
        # assert np.all(objs_i[rels_i[:, 0]] > 0) and np.all(objs_i[rels_i[:, 1]] > 0)

        if conf.return_top100:
            pred_entry = {
                'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE, #(23, 4) (16, 4)
                'pred_classes': objs_i, #(23,) (16,)
                'pred_rel_inds': rels_i_b100, #(506, 2) (240, 2)
                'obj_scores': obj_scores_i, #(23,) (16,)
                'rel_scores': pred_scores_i_b100,  # hack for now. (506, 51) (240, 51)
            }
        else:
            pred_entry = {
                'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE, #(23, 4) (16, 4)
                'pred_classes': objs_i, #(23,) (16,)
                'pred_rel_inds': rels_i, #(506, 2) (240, 2)
                'obj_scores': obj_scores_i, #(23,) (16,)
                'rel_scores': pred_scores_i,  # hack for now. (506, 51) (240, 51)
            }
        # pdb.set_trace()
        threshold = 0.35
        num_predicate = rel_scores_idx_b100.shape[0]
        keep_num = math.ceil(num_predicate*0.3) if num_predicate <= 100 else 30
        keep_idx_intop100 = list(range(int(keep_num)))

        bias_logit_norm = F.softmax(bias_logit[:, 1:], dim=1).data.cpu().numpy()
        mask_idxs_intop100 = np.nonzero(bias_logit_norm[rel_scores_idx_b100, :].max(1) < threshold)[0]
        mask_in_keep_num = np.where(mask_idxs_intop100<keep_num)[0].shape[0]
        mask_idxs_intop100 = mask_idxs_intop100.tolist()

        input_pred_num = len(mask_idxs_intop100) + keep_num - mask_in_keep_num
        input_pred_idxs_intop100 = np.zeros(input_pred_num, dtype=int)

        keep_idx_ininput = []
        mask_idx_ininput = []

        # pdb.set_trace()

        for i in range(input_pred_num):
            if i < keep_num:
                input_pred_idxs_intop100[i] = i
                if i not in mask_idxs_intop100:
                    keep_idx_ininput.append(i)
                else:
                    mask_idx_ininput.append(i)
            else:
                input_pred_idxs_intop100[i] = mask_idxs_intop100[i-keep_num+mask_in_keep_num]
                mask_idx_ininput.append(i)

        if len(mask_idx_ininput) == 0:
            mask_idx_ininput = None

        pred_entry['pred_rel_inds'] = rels_i_b100[input_pred_idxs_intop100]
        pred_entry['rel_scores'] = pred_scores_i_b100[input_pred_idxs_intop100]
        mask_idx = mask_idx_ininput

        # if batch_num == 32 or batch_num == 33:
        #     pdb.set_trace()

        pred_entry = glat_postprocess(pred_entry, if_predicting=True, mask_idx=mask_idx)
        pred_entry = cuda2numpy_dict(pred_entry)

        pred_scores_i_b100_updated = copy.deepcopy(pred_scores_i_b100)
        rels_i_b100_updated = copy.deepcopy(rels_i_b100)

        # pdb.set_trace()

        for i_ininput, i_intop100 in enumerate(input_pred_idxs_intop100):
            if mask_idx is not None:
                if i_ininput in mask_idx_ininput:
                    pred_scores_i_b100_updated[i_intop100] = pred_entry['rel_scores'][i_ininput, :-1]
                    rels_i_b100_updated[i_intop100] = pred_entry['pred_rel_inds'][i_ininput]

        pred_entry['rel_scores'] = pred_scores_i_b100_updated
        pred_entry['pred_rel_inds'] = rels_i_b100_updated

        if len(rels_i_a100.shape) == 1:
            pred_entry['rel_scores'] = pred_entry['rel_scores']
        else:
            pred_entry['pred_rel_inds'] = np.concatenate((pred_entry['pred_rel_inds'], rels_i_a100), axis=0)
            pred_entry['rel_scores'] = np.concatenate((pred_entry['rel_scores'], pred_scores_i_a100), axis=0)


        if mask_idx is not None:
            for idx in range(len(mask_idx)):
                sub = rels_i_b100.data[mask_idxs_intop100[idx], 0]
                obj = rels_i_b100.data[mask_idxs_intop100[idx], 1]
                pred_class = pred_entry['rel_scores'][mask_idxs_intop100[idx], 1:].argmax()+1
                if (int(sub), int(obj)) in dict_gt.keys():
                    accs[1] += 1
                    if int(pred_class) in dict_gt[(int(sub), int(obj))]:
                        accs[0] += 1

        eval_entry(conf.mode, gt_entry, pred_entry, evaluator, evaluator_multiple_preds,
                   evaluator_list, evaluator_multiple_preds_list)

print("Training starts now!")
# optimizer = get_optim(conf.lr * conf.num_gpus * conf.batch_size)

train_results = []
train_bias_logits = []
train_det_ress = []

eval_results = []
eval_logits = []
eval_det_ress = []

for epoch in range(start_epoch + 1, start_epoch + 1 + conf.num_epochs):
    print("start training epoch: ", epoch)
    scheduler.step()
    rez = train_epoch(epoch, train_results, train_bias_logits, train_det_ress)
    print("overall{:2d}: ({:.3f})\n{}".format(epoch, rez.mean(1)['total'], rez.mean(1)), flush=True)

    if use_tb:
        writer.add_scalar('loss/rel_loss', rez.mean(1)['rel_loss'], epoch)
        if conf.use_ggnn_obj:
            writer.add_scalar('loss/class_loss', rez.mean(1)['class_loss'], epoch)
        writer.add_scalar('loss/total', rez.mean(1)['total'], epoch)
    if conf.save_dir is not None:
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(), #{k:v for k,v in detector.state_dict().items() if not k.startswith('detector.')},
            # 'optimizer': optimizer.state_dict(),
            # 'scheduler': scheduler.state_dict(),
        }, os.path.join(conf.save_dir, '{}-{}.tar'.format('motifnet_glat', epoch)))

    recall, recall_mp, mean_recall, mean_recall_mp = val_epoch(epoch, eval_results, eval_logits, eval_det_ress)
    if use_tb:
        for key, value in recall.items():
            writer.add_scalar('eval_' + conf.mode + '_with_constraint/' + key, value, epoch)
        for key, value in recall_mp.items():
            writer.add_scalar('eval_' + conf.mode + '_without_constraint/' + key, value, epoch)
        for key, value in mean_recall.items():
            writer.add_scalar('eval_' + conf.mode + '_with_constraint/mean ' + key, value, epoch)
        for key, value in mean_recall_mp.items():
            writer.add_scalar('eval_' + conf.mode + '_without_constraint/mean ' + key, value, epoch)

