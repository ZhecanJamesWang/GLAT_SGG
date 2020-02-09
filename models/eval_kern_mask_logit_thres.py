from dataloaders.visual_genome import VGDataLoader, VG, build_graph_structure
import numpy as np
import torch

from config import ModelConfig
from lib.pytorch_misc import optimistic_restore
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list, eval_entry
from tqdm import tqdm
from config import BOX_SCALE, IM_SCALE
import dill as pkl
import os
from lib.kern_model import KERN
from lib.glat import GLATNET
from torch.autograd import Variable
from torch.nn import functional as F

import math
import copy

conf = ModelConfig()

train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                             use_proposals=conf.use_proposals,
                             filter_non_overlap=conf.mode == 'sgdet')

ind_to_predicates = train.ind_to_predicates  # ind_to_predicates[0] means no relationship
ind_to_classes = train.ind_to_classes

if conf.test:
    val = test

train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus)

detector = KERN(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                use_resnet=conf.use_resnet, use_proposals=conf.use_proposals,
                use_ggnn_obj=conf.use_ggnn_obj, ggnn_obj_time_step_num=conf.ggnn_obj_time_step_num,
                ggnn_obj_hidden_dim=conf.ggnn_obj_hidden_dim, ggnn_obj_output_dim=conf.ggnn_obj_output_dim,
                use_obj_knowledge=conf.use_obj_knowledge, obj_knowledge=conf.obj_knowledge,
                use_ggnn_rel=conf.use_ggnn_rel, ggnn_rel_time_step_num=conf.ggnn_rel_time_step_num,
                ggnn_rel_hidden_dim=conf.ggnn_rel_hidden_dim, ggnn_rel_output_dim=conf.ggnn_rel_output_dim,
                use_rel_knowledge=conf.use_rel_knowledge, rel_knowledge=conf.rel_knowledge,
                return_top100=conf.return_top100, return_unbias_logit=True)

model = GLATNET(vocab_num=[52, 153],
                feat_dim=300,
                nhid_glat_g=300,
                nhid_glat_l=300,
                nout=300,
                dropout=0.1,
                nheads=8,
                blank=152,
                types=[2] * 6)

detector.cuda()
ckpt = torch.load(conf.ckpt)

optimistic_restore(detector, ckpt['state_dict'])
# if conf.mode == 'sgdet':
#     det_ckpt = torch.load('checkpoints/new_vgdet/vg-19.tar')['state_dict']
#     detector.detector.bbox_fc.weight.data.copy_(det_ckpt['bbox_fc.weight'])
#     detector.detector.bbox_fc.bias.data.copy_(det_ckpt['bbox_fc.bias'])
#     detector.detector.score_fc.weight.data.copy_(det_ckpt['score_fc.weight'])
#     detector.detector.score_fc.bias.data.copy_(det_ckpt['score_fc.bias'])

# ckpt_glat = torch.load('/home/haoxuan/code/GBERT/models/2019-10-31-03-13_2_2_2_2_2_2_concat_no_init_mask/best_test_node_mask_predicate_acc')
# ckpt_glat = torch.load('/home/tangtangwzc/Common_sense/models/2019-11-03-17-51_2_2_2_2_2_2_concat_no_init_mask/best_test_node_mask_predicate_acc.pth')

# Pretrained Model
print('loading pretrained model')
ckpt_glat = torch.load('/home/tangtangwzc/Common_sense/models/2019-12-18-16-08_2_2_2_2_2_2_concat_no_init_mask/best_test_node_mask_predicate_acc.pth')
optimistic_restore(model, ckpt_glat['model'])
# model.load_state_dict(ckpt_glat['model'])
model.cuda()

model.eval()


def cuda2numpy_dict(dict):
    for key, value in dict.items():
        # pdb.set_trace()
        if torch.is_tensor(value):
            dict[key] = value.cpu().numpy()
        else:
            dict[key] = value.data.cpu().numpy()
    return dict


def numpy2cuda_dict(dict):
    for key, value in dict.items():
        dict[key] = torch.from_numpy(value).cuda()
    return dict


def tensor2variable(input):
    if torch.is_tensor(input):
        input = Variable(input)
    return input


def variable2tensor(input):
    if not torch.is_tensor(input):
        input = input.data
    return input

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
        # node_logit = total_data['node_logit'][i]
        # node_logit_pad = torch.Tensor([0] * node_logit.size()[0]).unsqueeze(-1).t()
        # node_logit = torch.cat((node_logit, Variable(node_logit_pad.t().cuda())), dim=1)

        # pad_node_logit = tensor2variable(torch.zeros((max_length - input_class.size(0)), node_logit.size()[1]).cuda())
        # node_logits.append(torch.cat((node_logit, pad_node_logit), 0).unsqueeze(0))

        pad_input_class = tensor2variable(blank_idx * torch.ones(max_length - input_class.size(0)).long().cuda())
        input_classes.append(torch.cat((input_class, pad_input_class), 0).unsqueeze(0))
        # input_classes.append(torch.cat((input_class, blank_idx*torch.ones(max_length-input_class.size(0)).long().cuda()), 0).unsqueeze(0))

        new_adj = torch.cat((adj, torch.zeros(max_length - adj.size(0), adj.size(1)).long().cuda()), 0)
        if max_length != new_adj.size(1):
            new_adj = torch.cat((new_adj, torch.zeros(new_adj.size(0), max_length - new_adj.size(1)).long().cuda()), 1)
        adjs.append(new_adj.unsqueeze(0))
        # pdb.set_trace()
        node_types.append(
            torch.cat((node_type, 2 * torch.ones(max_length - node_type.size(0)).long().cuda()), 0).unsqueeze(0))

    input_classes = torch.cat(input_classes, 0)

    adjs = torch.cat(adjs, 0)
    adjs_lbl = adjs
    adjs_con = torch.clamp(adjs, 0, 1)
    node_types = torch.cat(node_types, 0)

    return input_classes, adjs_con, adjs_lbl, node_types


# def my_collate(total_data):
#     blank_idx = 152
#     max_length = 0
#     sample_num = len(total_data['node_class'])
#     for i in range(sample_num):
#         max_length = max(max_length, total_data['node_class'][i].size(0))
#
#     input_classes = []
#     adjs = []
#     node_types = []
#     node_logits = []
#
#     for i in range(sample_num):
#         input_class = total_data['node_class'][i]
#         adj = total_data['adj'][i]
#         node_type = total_data['node_type'][i]
#         node_logit = total_data['node_logit'][i]
#         node_logit_pad = torch.Tensor([0] * node_logit.size()[0]).unsqueeze(-1).t()
#         node_logit = torch.cat((node_logit, Variable(node_logit_pad.t().cuda())), dim=1)
#
#         pad_node_logit = tensor2variable(torch.zeros((max_length - input_class.size(0)), node_logit.size()[1]).cuda())
#         node_logits.append(torch.cat((node_logit, pad_node_logit), 0).unsqueeze(0))
#
#         pad_input_class = tensor2variable(blank_idx * torch.ones(max_length - input_class.size(0)).long().cuda())
#         input_classes.append(torch.cat((input_class, pad_input_class), 0).unsqueeze(0))
#         # input_classes.append(torch.cat((input_class, blank_idx*torch.ones(max_length-input_class.size(0)).long().cuda()), 0).unsqueeze(0))
#
#         new_adj = torch.cat((adj, torch.zeros(max_length - adj.size(0), adj.size(1)).long().cuda()), 0)
#         if max_length != new_adj.size(1):
#             new_adj = torch.cat((new_adj, torch.zeros(new_adj.size(0), max_length - new_adj.size(1)).long().cuda()), 1)
#         adjs.append(new_adj.unsqueeze(0))
#         # pdb.set_trace()
#         node_types.append(
#             torch.cat((node_type, 2 * torch.ones(max_length - node_type.size(0)).long().cuda()), 0).unsqueeze(0))
#
#     input_classes = torch.cat(input_classes, 0)
#
#     node_logits = torch.cat(node_logits, 0)
#
#     adjs = torch.cat(adjs, 0)
#     adjs_lbl = adjs
#     adjs_con = torch.clamp(adjs, 0, 1)
#     node_types = torch.cat(node_types, 0)
#     return input_classes, adjs_con, adjs_lbl, node_types, node_logits


def glat_wrapper(total_data):
    # Batch size assumed to be 1
    # input_class, adjs_con, adjs_lbl, node_type, node_logit = my_collate(total_data)
    input_class, adjs_con, adjs_lbl, node_type = my_collate(total_data)

    if torch.is_tensor(input_class):
        input_class = Variable(input_class)
    # if torch.is_tensor(node_logit):
    #     node_logit = Variable(node_logit)
    if not torch.is_tensor(node_type):
        node_type = node_type.data
    if torch.is_tensor(adjs_con):
        adj_con = Variable(adjs_con)

    pred_label, pred_connect = model(input_class, adj_con, node_type)
    # pred_label, pred_connect = model(input_class, adj_con, node_type, node_logit)

    # pred_label_predicate = input_class[node_type == 0]
    # pred_label_entities = input_class[node_type == 1]

    pred_label_predicate = pred_label[0]  # flatten predicate (B*N, 51)
    pred_label_entities = pred_label[1]  # flatten entities

    return pred_label_predicate, pred_label_entities
    # return pred_label_predicate.data.cpu().numpy(), pred_label_entities.data.cpu().numpy()


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
    pred_entry['pred_relations'] = torch.cat((pred_entry['pred_rel_inds'], pred_entry['rel_classes']), dim=1)

    total_data = build_graph_structure(pred_entry, ind_to_classes, ind_to_predicates, if_predicting=if_predicting)

    pred_label_predicate, pred_label_entities = glat_wrapper(total_data)
    pred_entry['rel_scores'] = pred_label_predicate

    return pred_entry


all_pred_entries = []


def val_batch(batch_num, b, evaluator,evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list, accs):

    dict_gt, bias_logit, det_res = detector[b]

    if conf.num_gpus == 1:
        det_res = [det_res]

    for i, det in enumerate(det_res):

        if len(det) == 5 and not conf.return_top100:
            (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) = det
            rels_i_a100 = np.asarray([])
        else:
            (boxes_i, objs_i, obj_scores_i, rels_i_b100, pred_scores_i_b100, rels_i_a100, pred_scores_i_a100,
             rel_scores_idx_b100, rel_scores_idx_a100) = det

        # print("boxes_i.size(): ", boxes_i.shape)
        # print("rels_i_b100.size(): ", rels_i_b100.shape)
        # print("rels_i_a100.size(): ", rels_i_a100.shape)

        gt_entry = {
            'gt_classes': val.gt_classes[batch_num + i].copy(),  # (23,) (16,)
            'gt_relations': val.relationships[batch_num + i].copy(),  # (29, 3) (6, 3)
            'gt_boxes': val.gt_boxes[batch_num + i].copy(),  # (23, 4) (16, 4)
        }

        if conf.return_top100:
            pred_entry = {
                'pred_boxes': boxes_i * BOX_SCALE / IM_SCALE,  # (23, 4) (16, 4)
                'pred_classes': objs_i,  # (23,) (16,)
                'pred_rel_inds': rels_i_b100,  # (506, 2) (240, 2)
                'obj_scores': obj_scores_i,  # (23,) (16,)
                'rel_scores': pred_scores_i_b100,  # hack for now. (506, 51) (240, 51)
            }
        else:
            pred_entry = {
                'pred_boxes': boxes_i * BOX_SCALE / IM_SCALE,  # (23, 4) (16, 4)
                'pred_classes': objs_i,  # (23,) (16,)
                'pred_rel_inds': rels_i,  # (506, 2) (240, 2)
                'obj_scores': obj_scores_i,  # (23,) (16,)
                'rel_scores': pred_scores_i,  # hack for now. (506, 51) (240, 51)
            }

        global threshold
        # threshold = 0.35
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
                number_mask[0] += 1
                sub = rels_i_b100.data[mask_idxs_intop100[idx], 0]
                obj = rels_i_b100.data[mask_idxs_intop100[idx], 1]
                pred_class = pred_entry['rel_scores'][mask_idxs_intop100[idx], 1:].argmax()+1
                if (int(sub), int(obj)) in dict_gt.keys():
                    accs[1] += 1
                    if int(pred_class) in dict_gt[(int(sub), int(obj))]:
                        accs[0] += 1


        all_pred_entries.append(pred_entry)

        eval_entry(conf.mode, gt_entry, pred_entry, evaluator, evaluator_multiple_preds,
                   evaluator_list, evaluator_multiple_preds_list)


evaluator = BasicSceneGraphEvaluator.all_modes()
evaluator_multiple_preds = BasicSceneGraphEvaluator.all_modes(multiple_preds=True)
evaluator_list = []  # for calculating recall of each relationship except no relationship
evaluator_multiple_preds_list = []
for index, name in enumerate(ind_to_predicates):
    if index == 0:
        continue
    evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))
    evaluator_multiple_preds_list.append((index, name, BasicSceneGraphEvaluator.all_modes(multiple_preds=True)))

# if conf.cache is not None and os.path.exists(conf.cache):
#     print("Found {}! Loading from it".format(conf.cache))
#     with open(conf.cache, 'rb') as f:
#         all_pred_entries = pkl.load(f)
#     for i, pred_entry in enumerate(tqdm(all_pred_entries)):
#         gt_entry = {
#             'gt_classes': val.gt_classes[i].copy(),
#             'gt_relations': val.relationships[i].copy(),
#             'gt_boxes': val.gt_boxes[i].copy(),
#         }
#
#         eval_entry(conf.mode, gt_entry, pred_entry, evaluator, evaluator_multiple_preds,
#                    evaluator_list, evaluator_multiple_preds_list)
#
#     recall = evaluator[conf.mode].print_stats()
#     recall_mp = evaluator_multiple_preds[conf.mode].print_stats()
#
#     mean_recall = calculate_mR_from_evaluator_list(evaluator_list, conf.mode, save_file=conf.save_rel_recall)
#     mean_recall_mp = calculate_mR_from_evaluator_list(evaluator_multiple_preds_list, conf.mode, multiple_preds=True,
#                                                       save_file=conf.save_rel_recall)
#
# else:
detector.eval()
accs = [0, 0]
number_mask = [0]

threshold = conf.threshold
print('Threshold is', threshold)

for val_b, batch in enumerate(tqdm(val_loader)):
    val_batch(conf.num_gpus * val_b, batch, evaluator, evaluator_multiple_preds, evaluator_list,
              evaluator_multiple_preds_list, accs)
    torch.cuda.empty_cache()

print('test acc of mask:', accs[0] * 1.0 / accs[1])
print('number of mask:', accs[1])
print('total number of mask:', number_mask)


recall = evaluator[conf.mode].print_stats()
recall_mp = evaluator_multiple_preds[conf.mode].print_stats()

mean_recall = calculate_mR_from_evaluator_list(evaluator_list, conf.mode, save_file=conf.save_rel_recall)
mean_recall_mp = calculate_mR_from_evaluator_list(evaluator_multiple_preds_list, conf.mode, multiple_preds=True,
                                                  save_file=conf.save_rel_recall)

torch.cuda.empty_cache()

if conf.cache is not None:
    with open(conf.cache, 'wb') as f:
        pkl.dump(all_pred_entries, f)
