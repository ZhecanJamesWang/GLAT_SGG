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
from lib.glat import GLATNET
import pdb
from torch.autograd import Variable
import copy
from scipy.special import softmax
import torch.optim.lr_scheduler as lr_scheduler

#--------updated--------
import sys
import os
import matplotlib.pyplot as plt

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
                    return_top100=True
                    )


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
    # ckpt_glat = torch.load(
    #     '/home/tangtangwzc/Common_sense/models/2019-11-03-17-28_2_2_2_2_2_2_concat_no_init_mask/best_test_node_mask_predicate_acc.pth')
    ckpt_glat = torch.load('/home/haoxuan/code/KERN/checkpoints/motifnet_glat_mask_pred_sgcls/motifnet_glat-22.tar')
    # ---------------pretrained model mask ration 0.7
    # ckpt_glat = torch.load('/home/tangtangwzc/Common_sense/models/2019-11-07-23-50_2_2_2_2_2_2_concat_no_init_mask/best_test_node_mask_predicate_acc.pth')

    # optimistic_restore(model, ckpt_glat['model'])
    optimistic_restore(model, ckpt_glat['state_dict'])
    model.cuda()
    start_epoch = -1
    optimizer, scheduler = get_optim(conf.lr, last_epoch=start_epoch)


def val_epoch():

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
    for val_b, batch in enumerate(val_loader):
        val_batch(conf.num_gpus * val_b, batch, evaluator, evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list, accs)

    recall = evaluator[conf.mode].print_stats()
    recall_mp = evaluator_multiple_preds[conf.mode].print_stats()

    mean_recall = calculate_mR_from_evaluator_list(evaluator_list, conf.mode)
    mean_recall_mp = calculate_mR_from_evaluator_list(evaluator_multiple_preds_list, conf.mode, multiple_preds=True)

    print('test acc of mask:', accs[0] * 1.0 / accs[1])
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

    total_data = build_graph_structure(pred_entry, ind_to_classes, ind_to_predicates, if_predicting=if_predicting)

    pred_label_predicate, pred_label_entities = glat_wrapper(total_data)
    pred_entry['rel_scores'] = pred_label_predicate

    # =====================================
    # if if_predicting:
    #     pred_entry = cuda2numpy_dict(pred_entry)
    #     obj_scores0 = pred_entry['obj_scores'][pred_entry['pred_rel_inds'][:, 0]]
    #     obj_scores1 = pred_entry['obj_scores'][pred_entry['pred_rel_inds'][:, 1]]
    #
    #     pred_scores_max = np.max(pred_entry['rel_scores'][:, 1:], axis=1)
    #
    #     rel_scores_argmaxed = pred_scores_max * obj_scores0 * obj_scores1
    #     rel_scores_idx = np.argsort(rel_scores_argmaxed, axis=0)[::-1]
    #
    #     pred_entry['rel_scores'] = pred_entry['rel_scores'][rel_scores_idx]

    # # predicate_list = []
    # for i in range(len(gt_entry['gt_relations'])):
    #     subj_idx = gt_entry['gt_relations'][i][0]
    #     subj_class_idx = gt_entry['gt_classes'][subj_idx]
    #
    #     obj_idx = gt_entry['gt_relations'][i][1]
    #     obj_class_idx = gt_entry['gt_classes'][obj_idx]
    #
    #     predicate_idx = gt_entry['gt_relations'][i][2]
    #     predicate = ind_to_predicates[predicate_idx]
    #
    #     subj = ind_to_classes[subj_class_idx]
    #     obj = ind_to_classes[obj_class_idx]
    #
    #     print(subj)
    #     print(predicate)
    #     print(obj)
    #
    #     # predicate_list.append(predicate)


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


def val_batch(batch_num, b, evaluator, evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list, accs):
    dict_gt, det_res = detector[b]


    if conf.num_gpus == 1:
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

        pdb.set_trace()

        wrong_idxs = []
        right_idxs = []
        for j in range(len(rels_i_b100)):
            if (int(rels_i_b100[j][0]), int(rels_i_b100[j][1])) in dict_gt:
                pred_lbl = pred_scores_i_b100[j, 1:].argmax(0) + 1
                if int(pred_lbl) not in dict_gt[(int(rels_i_b100[j][0]), int(rels_i_b100[j][1]))]:
                    wrong_idxs.append(j)
                else:
                    right_idxs.append(j)

                num_classes = pred_scores_i_b100.shape[1]
                color = ['b']*num_classes
                for gt_class in dict_gt[(int(rels_i_b100[j][0]), int(rels_i_b100[j][1]))]:
                    color[gt_class] = 'r'
                plt.bar(range(num_classes), list(pred_scores_i_b100[j]), color=color)
                plt.grid(True)
                if int(pred_lbl) not in dict_gt[(int(rels_i_b100[j][0]), int(rels_i_b100[j][1]))]:
                    plt.savefig('./vis_result/wrong_{}_{}.jpg'.format(batch_num, j))
                else:
                    plt.savefig('./vis_result/right_{}_{}.jpg'.format(batch_num, j))
                plt.close()


        if len(wrong_idxs) == 0:
            mask_idx = None
        else:
            mask_idx = wrong_idxs

        print('right prediction in top100:', len(right_idxs))
        print('right idx', right_idxs)

        print('wrong prediction in top100:', len(wrong_idxs))
        print('wrong idx', wrong_idxs)

        print('all ground truths:', len(dict_gt))

        # pdb.set_trace()

        # gt_label = gt_rel_labels[:, -1][rel_scores_idx_b100]
        # pdb.set_trace()
        # non_background_idx = torch.nonzero(gt_label)
        # non_background_idx = non_background_idx.squeeze()
        # pred_label = pred_entry['rel_scores'][:, 1:].max(1)[1] + 1
        # pred_label_no_back = pred_label[non_background_idx]
        # gt_label_no_back = gt_label[non_background_idx]
        #
        # wrong_idx = torch.nonzero(pred_label_no_back != gt_label_no_back)
        # if len(wrong_idx) != 0:
        #     wrong_idx = wrong_idx.squeeze(-1)
        #     mask_idx = non_background_idx[wrong_idx]
        # else:
        #     mask_idx = None

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        pred_entry = glat_postprocess(pred_entry, if_predicting=True, mask_idx=mask_idx)
        pred_entry = cuda2numpy_dict(pred_entry)

        if len(rels_i_a100.shape) == 1:
            pred_entry['rel_scores'] = pred_entry['rel_scores'][:, :-1]
        else:
            pred_entry['pred_rel_inds'] = np.concatenate((pred_entry['pred_rel_inds'], rels_i_a100), axis=0)
            pred_entry['rel_scores'] = np.concatenate((pred_entry['rel_scores'][:, :-1], pred_scores_i_a100), axis=0)

        correct_sample_num = 0
        if mask_idx is not None:
            for idx in mask_idx:
                accs[1] += 1
                sub = rels_i_b100.data[idx, 0]
                obj = rels_i_b100.data[idx, 1]
                pred_class = pred_entry['rel_scores'][idx, 1:].argmax()+1
                if int(pred_class) in dict_gt[(int(sub), int(obj))]:
                    accs[0] += 1
                    correct_sample_num += 1

        print('corrected samples:', correct_sample_num)

        eval_entry(conf.mode, gt_entry, pred_entry, evaluator, evaluator_multiple_preds,
                   evaluator_list, evaluator_multiple_preds_list)

print("Visualization starts now!")
# optimizer = get_optim(conf.lr * conf.num_gpus * conf.batch_size)

# for epoch in range(start_epoch + 1, start_epoch + 1 + conf.num_epochs):
    # print("start training epoch: ", epoch)
    # scheduler.step()
    # # pdb.set_trace()
    # rez = train_epoch(epoch)
    # print("overall{:2d}: ({:.3f})\n{}".format(epoch, rez.mean(1)['total'], rez.mean(1)), flush=True)
    #
    # if use_tb:
    #     writer.add_scalar('loss/rel_loss', rez.mean(1)['rel_loss'], epoch)
    #     if conf.use_ggnn_obj:
    #         writer.add_scalar('loss/class_loss', rez.mean(1)['class_loss'], epoch)
    #     writer.add_scalar('loss/total', rez.mean(1)['total'], epoch)
    # if conf.save_dir is not None:
    #     torch.save({
    #         'epoch': epoch,
    #         'state_dict': model.state_dict(), #{k:v for k,v in detector.state_dict().items() if not k.startswith('detector.')},
    #         # 'optimizer': optimizer.state_dict(),
    #         # 'scheduler': scheduler.state_dict(),
    #     }, os.path.join(conf.save_dir, '{}-{}.tar'.format('motifnet_glat', epoch)))

recall, recall_mp, mean_recall, mean_recall_mp = val_epoch()
# if use_tb:
#     for key, value in recall.items():
#         writer.add_scalar('eval_' + conf.mode + '_with_constraint/' + key, value, 0)
#     for key, value in recall_mp.items():
#         writer.add_scalar('eval_' + conf.mode + '_without_constraint/' + key, value, 0)
#     for key, value in mean_recall.items():
#         writer.add_scalar('eval_' + conf.mode + '_with_constraint/mean ' + key, value, 0)
#     for key, value in mean_recall_mp.items():
#         writer.add_scalar('eval_' + conf.mode + '_without_constraint/mean ' + key, value, 0)

