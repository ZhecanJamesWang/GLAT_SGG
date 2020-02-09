
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
from lib.glat import GLATNET
from torch.autograd import Variable
import pdb
from torch.nn import functional as F


conf = ModelConfig()
if conf.model_s_m == 'motifnet':
    from lib.motifnet_model import RelModel
elif conf.model_s_m == 'stanford':
    from lib.stanford_model import RelModelStanford as RelModel
else:
    raise ValueError()


train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                          use_proposals=conf.use_proposals,
                          filter_non_overlap=conf.mode == 'sgdet')
if conf.test:
    val = test
train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus)

ind_to_predicates = train.ind_to_predicates # ind_to_predicates[0] means no relationship
ind_to_classes = train.ind_to_classes


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
                    limit_vision=limit_vision,
                    return_top100=True,
                    return_unbias_logit=True,
                    )

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
if conf.model_s_m == 'stanford':
    print('stanford mode ckpt wrong!!!')
    # ckpt_glat = torch.load('/home/haoxuan/code/KERN/checkpoints/stanford_glat_1/stanford_glat-20.tar')

    # Pretrained Model
    ckpt_glat = torch.load(
        '/home/tangtangwzc/Common_sense/models/2019-12-18-16-08_2_2_2_2_2_2_concat_no_init_mask/best_test_node_mask_predicate_acc.pth')

    # ckpt_glat = torch.load(
    #     '/home/tangtangwzc/Common_sense/models/2019-11-03-17-28_2_2_2_2_2_2_concat_no_init_mask/best_test_node_mask_predicate_acc.pth')

elif conf.model_s_m == 'motifnet':
    # Finetuned model
    # ckpt_glat = torch.load('/home/haoxuan/code/KERN/checkpoints/motifnet_glat_mask_pred/motifnet_glat-49.tar')

    # Pretrained Model
    ckpt_glat = torch.load(
        '/home/tangtangwzc/Common_sense/models/2019-12-18-16-08_2_2_2_2_2_2_concat_no_init_mask/best_test_node_mask_predicate_acc.pth')

    # ckpt_glat = torch.load(
    #     '/home/tangtangwzc/Common_sense/models/2019-11-03-17-28_2_2_2_2_2_2_concat_no_init_mask/best_test_node_mask_predicate_acc.pth')

# Finetuned model
# optimistic_restore(model, ckpt_glat['state_dict'])

# Pretrained Model
optimistic_restore(model, ckpt_glat['model'])
print('finish pretrained loading')
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
    # node_logits = []

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

    # node_logits = torch.cat(node_logits, 0)

    adjs = torch.cat(adjs, 0)
    adjs_lbl = adjs
    adjs_con = torch.clamp(adjs, 0, 1)
    node_types = torch.cat(node_types, 0)

    # return input_classes, adjs_con, adjs_lbl, node_types, node_logits
    return input_classes, adjs_con, adjs_lbl, node_types


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

def val_batch(batch_num, b, evaluator,evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list):
    # det_res = detector[b]
    # dict_gt, det_res = detector[b]
    # dict_gt, unbias_logit, det_res = detector[b]
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

        num_predicate = rel_scores_idx_b100.shape[0]

        # V1 Ranking
        # for i, ranking in enumerate(rankings):
        #     for j in range(int(num_predicate * ranking), num_predicate):
        #         sub = rels_i_b100.data[j, 0]
        #         obj = rels_i_b100.data[j, 1]
        #         pred_class = pred_entry['rel_scores'][j, 1:].argmax()+1
        #         if (int(sub), int(obj)) in dict_gt.keys():
        #             if int(pred_class) in dict_gt[(int(sub), int(obj))]:
        #                 pred_right[i] += 1
        #             else:
        #                 pred_wrong[i] += 1

        # V2 Threshold V6 logit threshold
        # global total_pred_num
        # global total_num_underthres
        # global total_num_cover_gt_pair
        # # unbias_logit = F.softmax(unbias_logit[:, 1:], dim=1).data.cpu().numpy()
        # bias_logit = F.softmax(bias_logit[:, 1:], dim=1).data.cpu().numpy()
        # # topk = int(pred_scores_i_b100.shape[0] * 0.3) if pred_scores_i_b100.shape[0]<100 else 30
        # topk = int(pred_scores_i_b100.shape[0])
        # total_pred_num += topk
        # topk_pred_scores = bias_logit[rel_scores_idx_b100[:topk]]
        # for i, threshold in enumerate(rankings):
        #     pred_idxs = np.nonzero(topk_pred_scores.max(1) < threshold)[0]
        #     pred_idxs = pred_idxs.tolist()
        #     total_num_underthres[i] += len(pred_idxs)
        #     for j in pred_idxs:
        #         sub = rels_i_b100.data[j, 0]
        #         obj = rels_i_b100.data[j, 1]
        #         pred_class_bias = pred_entry['rel_scores'][j, 1:].argmax()+1
        #         # pred_class_unbias = topk_pred_scores[j].argmax()
        #         if (int(sub), int(obj)) in dict_gt.keys():
        #             if int(pred_class_bias) in dict_gt[(int(sub), int(obj))]:
        #                 pred_right[i] += 1
        #             else:
        #                 pred_wrong[i] += 1
        #             total_num_cover_gt_pair[i] +=1


        # V7 certain threshold with different range of top sampels
        global threshold
        global total_pred_num
        # global total_num_underthres
        global total_num_cover_gt_pair
        bias_logit = F.softmax(bias_logit[:, 1:], dim=1).data.cpu().numpy()
        pred_num = int(pred_scores_i_b100.shape[0])

        # pdb.set_trace()
        for i in range(len(total_pred_num)):
            total_pred_num[i] += int(pred_num * topn[i]) if i == 0 else int(pred_num * topn[i]) - int(pred_num * topn[i-1])
        bias_logit_b100 = bias_logit[rel_scores_idx_b100]

        pred_idxs = np.nonzero(bias_logit_b100.max(1) < threshold)[0]
        pred_idxs = pred_idxs.tolist()

        # pdb.set_trace()

        for j in pred_idxs:
            sub = rels_i_b100.data[j, 0]
            obj = rels_i_b100.data[j, 1]
            pred_class_bias = pred_entry['rel_scores'][j, 1:].argmax()+1

            low_range = [int((i-0.1)*pred_num) for i in topn]
            upper_range = [int(i*pred_num) for i in topn]
            for i in range(len(upper_range)):
                low = low_range[i]
                upper = upper_range[i]
                if j>= low and j < upper:
                    save_range = i

            # if batch_num == 356 or batch_num == 357:
            #     pdb.set_trace()

            if (int(sub), int(obj)) in dict_gt.keys():
                if int(pred_class_bias) in dict_gt[(int(sub), int(obj))]:
                    pred_right[save_range] += 1
                else:
                    pred_wrong[save_range] += 1
                total_num_cover_gt_pair[save_range] +=1

        # pdb.set_trace()


        # V3/V4 threshold + co-occurance matrix
        # for i, threshold in enumerate(rankings):
        #     pred_idxs = np.nonzero(pred_scores_i_b100[:, 1:].max(1) < threshold)[0]
        #     pred_idxs = pred_idxs.tolist()
        #     for pred_idx in pred_idxs:
        #         sub = int(rels_i_b100.data[pred_idx, 0])
        #         sub_class = int(objs_i[sub])
        #         obj = int(rels_i_b100.data[pred_idx, 1])
        #         obj_class = int(objs_i[obj])
        #         pred_class = int(pred_entry['rel_scores'][pred_idx, 1:].argmax()+1)
        #
        #         # pdb.set_trace()
        #
        #         if (sub, obj) in dict_gt.keys():
        #             if pred_class in dict_gt[(sub, obj)]:
        #                 if pred_class in np.argsort(rel_matrix[sub_class, obj_class, 1:])[::-1][:2]+1:
        #                     pred_right_top2[i] += 1
        #                 elif pred_class in np.nonzero(rel_matrix[sub_class, obj_class])[0]:
        #                     pred_right_others2[i] += 1
        #                 elif pred_class not in np.nonzero(rel_matrix[sub_class, obj_class])[0]:
        #                     pred_right_notinco[i] += 1
        #             else:
        #                 if pred_class in np.argsort(rel_matrix[sub_class, obj_class, 1:])[::-1][:2]+1:
        #                     pred_wrong_top2[i] += 1
        #                 elif pred_class in np.nonzero(rel_matrix[sub_class, obj_class])[0]:
        #                     pred_wrong_others2[i] += 1
        #                 elif pred_class not in np.nonzero(rel_matrix[sub_class, obj_class])[0]:
        #                     pred_wrong_notinco[i] += 1
        #
        #         if rel_matrix[sub_class, obj_class].max() != 1:
        #             pred_exist_based_on_mat[i] += 1
        #             if (sub, obj) in dict_gt.keys():
        #                 pred_exist_based_on_mat_ingt[i] += 1
        #         else:
        #             pred_not_exist_based_on_mat[i] += 1
        #             if (sub, obj) in dict_gt.keys():
        #                 pred_not_exist_based_on_mat_butreal[i] += 1

        # V5 Precision and Recall in topn
        # global total_gt
        # global total_correct
        # global total_sample
        # global total_gt_pairs
        # global total_bg_pairs
        # global total_bg_erasable_pairs
        # total_gt += len(dict_gt.keys())
        # num_predicate = rel_scores_idx_b100.shape[0]
        # for i in range(len(topns)):
        #     if i*10 >= num_predicate:
        #         break
        #     topn_idx = list(range(i*10, (i+1)*10)) if num_predicate - (i+1)*10 >= 10 else list(range(i*10, num_predicate))
        #     total_sample[i] += i*10 + len(topn_idx)
        #     for pred_idx in topn_idx:
        #         sub = int(rels_i_b100.data[pred_idx, 0])
        #         sub_class = int(objs_i[sub])
        #         obj = int(rels_i_b100.data[pred_idx, 1])
        #         obj_class = int(objs_i[obj])
        #         pred_class = int(pred_entry['rel_scores'][pred_idx, 1:].argmax()+1)
        #         if (sub, obj) in dict_gt.keys():
        #             total_gt_pairs = [gt_pair+1 if idx >= i else gt_pair for idx, gt_pair in enumerate(total_gt_pairs)]
        #             if pred_class in dict_gt[(sub, obj)]:
        #                 total_correct = [cor+1 if idx >= i else cor for idx, cor in enumerate(total_correct)]
        #         else:
        #             total_bg_pairs = [bg_pair+1 if idx >= i else bg_pair for idx, bg_pair in enumerate(total_bg_pairs)]
        #             if pred_class not in np.nonzero(rel_matrix[sub_class, obj_class])[0]:
        #                 total_bg_erasable_pairs = [bg_er_pair+1 if idx >= i else bg_er_pair for idx, bg_er_pair in enumerate(total_bg_erasable_pairs)]


        # num_predicate = rel_scores_idx_b100.shape[0]
        # mask_idx = torch.Tensor(range(int(num_predicate*(1-0.3)),num_predicate)).long().cuda()
        # if len(mask_idx) == 0:
        #     mask_idx = None
        #
        # pred_entry = glat_postprocess(pred_entry, if_predicting=True, mask_idx=mask_idx)
        # pred_entry = cuda2numpy_dict(pred_entry)
        #
        # if len(rels_i_a100.shape) == 1:
        #     pred_entry['rel_scores'] = pred_entry['rel_scores'][:, :-1]
        # else:
        #
        #     pred_entry['pred_rel_inds'] = np.concatenate((pred_entry['pred_rel_inds'], rels_i_a100), axis=0)
        #     pred_entry['rel_scores'] = np.concatenate((pred_entry['rel_scores'][:, :-1], pred_scores_i_a100), axis=0)
        #
        # pdb.set_trace()
        # if mask_idx is not None:
        #     for idx in range(mask_idx.size(0)):
        #         sub = rels_i_b100.data[mask_idx[idx], 0]
        #         obj = rels_i_b100.data[mask_idx[idx], 1]
        #         pred_class = pred_entry['rel_scores'][mask_idx[idx], 1:].argmax()+1
        #         if (int(sub), int(obj)) in dict_gt.keys():
        #             accs[1] += 1
        #             if int(pred_class) in dict_gt[(int(sub), int(obj))]:
        #                 accs[0] += 1
        #
        # all_pred_entries.append(pred_entry)
        #
        # eval_entry(conf.mode, gt_entry, pred_entry, evaluator, evaluator_multiple_preds,
        #            evaluator_list, evaluator_multiple_preds_list)


evaluator = BasicSceneGraphEvaluator.all_modes()
evaluator_multiple_preds = BasicSceneGraphEvaluator.all_modes(multiple_preds=True)
evaluator_list = []  # for calculating recall of each relationship except no relationship
evaluator_multiple_preds_list = []
# evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=False)
for index, name in enumerate(ind_to_predicates):
    if index == 0:
        continue
    evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))
    evaluator_multiple_preds_list.append((index, name, BasicSceneGraphEvaluator.all_modes(multiple_preds=True)))

detector.eval()
# V1 rankings
# rankings = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

# V2 threshold +
# rankings = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
# rankings = [0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# rankings = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# print('ranking is:', rankings)
# pred_right = [0] * 10

# V6 logit threshold
# rankings = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# print('ranking is:', rankings)
# pred_right = [0] * 10
# total_pred_num = 0
# total_num_underthres = [0] * 10
# total_num_cover_gt_pair = [0] * 10
# pred_wrong = [0] * 10

# V7 certain threshold with different range of top sampels
threshold = 0.35
print('threshold is:', threshold)
topn = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
print('topn is:', topn)
pred_right = [0] * 10
total_pred_num = [0] * 10
total_num_cover_gt_pair = [0] * 10
pred_wrong = [0] * 10

# V3&V4 threshold + co-occurance matrix (v3 top1 v4 top2)
# rel_matrix = np.load('./prior_matrices/rel_matrix.npy')
# rankings = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
# pred_right_top2 = [0] * len(rankings)
# pred_right_others2 = [0] * len(rankings)
# pred_right_notinco = [0] * len(rankings)
# pred_wrong_top2 = [0] * len(rankings)
# pred_wrong_others2 = [0] * len(rankings)
# pred_wrong_notinco = [0] * len(rankings)
# pred_exist_based_on_mat = [0] * len(rankings)
# pred_exist_based_on_mat_ingt = [0] * len(rankings)
# pred_not_exist_based_on_mat = [0] * len(rankings)
# pred_not_exist_based_on_mat_butreal = [0] * len(rankings)

# V5 Precision and Recall in topn
# rel_matrix = np.load('./prior_matrices/rel_matrix.npy')
# topns = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# total_correct = [0]*10
# total_gt_pairs = [0] * 10
# total_bg_pairs = [0] * 10
# total_bg_erasable_pairs = [0] * 10
# total_sample = [0]*10
# total_gt = 0

print('Start over testing set')
for val_b, batch in enumerate(tqdm(val_loader)):
    # val_batch(conf.num_gpus*val_b, batch, evaluator)
    val_batch(conf.num_gpus * val_b, batch, evaluator, evaluator_multiple_preds, evaluator_list,
              evaluator_multiple_preds_list)

    torch.cuda.empty_cache()
    if val_b % 500 == 0 and val_b != 0:
        # V1 or V2 or V6
        # print('number of right prediction:', pred_right)
        # print('ratio of right prediction:', [pred_right[i]*1.0/(pred_right[i]+pred_wrong[i]) for i in range(len(pred_right))])
        # print('\n')
        # print('number of wrong prediction:', pred_wrong)
        # print('ratio of wrong prediction:', [pred_wrong[i]*1.0/(pred_right[i]+pred_wrong[i]) for i in range(len(pred_wrong))])
        # print('\n')
        # print('number of total predication:', total_pred_num)
        # print('number of total under threshold:', total_num_underthres)
        # print('number of total cover gt:', total_num_cover_gt_pair)
        # print('-----------')
        # print('\n')

        # V7
        print('number of right prediction:', pred_right)
        print('ratio of right prediction:', [pred_right[i]*1.0/(pred_right[i]+pred_wrong[i]) for i in range(len(pred_right))])
        print('\n')
        print('number of wrong prediction:', pred_wrong)
        print('ratio of wrong prediction:', [pred_wrong[i]*1.0/(pred_right[i]+pred_wrong[i]) for i in range(len(pred_wrong))])
        print('\n')
        print('number of total predication:', total_pred_num)
        print('number of total cover gt:', total_num_cover_gt_pair)
        print('ratio of cover gt and total predication :', [total_num_cover_gt_pair[i]/total_pred_num[i] for i in range(len(total_pred_num))])
        print('-----------')
        print('\n')

        # V3/v4
        # print('\n')
        # print('>>>>>>>>>>>>>>>>>>')
        # print('Right predicate & in top2:', pred_right_top2)
        # print('Right predicate & in others:', pred_right_others2)
        # print('Right predicate & not in available classes:', pred_right_notinco)
        # print('\n')
        # print('Wrong predicate & in top2:', pred_wrong_top2)
        # print('Wrong predicate & in others:', pred_wrong_others2)
        # print('Wrong predicate & not in available classes:', pred_wrong_notinco)
        # print('\n')
        # print('Predicate exist based on mat:', pred_exist_based_on_mat)
        # print('Predicate exist & is gt based on mat:', pred_exist_based_on_mat_ingt)
        # print('Predicate not exist based on mat:', pred_not_exist_based_on_mat)
        # print('Predicate not exist & but gt based on mat:', pred_not_exist_based_on_mat_butreal)
        # print('<<<<<<<<<<<<<<<<<')

        # V5
        # print('\n')
        # print('>>>>>>>>>>>>>>>>>>')
        # print('total_correct number:', total_correct)
        # print('total_gt_pairs number:', total_gt_pairs)
        # print('total_bg_pairs number:', total_bg_pairs)
        # print('total_bg_erasable_pairs number:', total_bg_erasable_pairs)
        # print('total_sample number:', total_sample)
        # print('total_gt number:', total_gt)
        # print('<<<<<<<<<<<<<<<<<')

# V1 or V2
print('Finished all testing set')
print('number of right prediction and ratio:', pred_right)
print('ratio of right prediction and ratio: ',
      [pred_right[i] * 1.0 / (pred_right[i] + pred_wrong[i]) for i in range(len(pred_right))])
print('\n')
print('number of wrong prediction:', pred_wrong)
print('ratio of wrong prediction:',
      [pred_wrong[i] * 1.0 / (pred_right[i] + pred_wrong[i]) for i in range(len(pred_wrong))])
print('number of total predication:', total_pred_num)
print('number of total cover gt:', total_num_cover_gt_pair)

print('total ratio of right and wrong predication:', sum(pred_right)/sum(total_num_cover_gt_pair))
print('total ratio of cover gt and total predication :',
      [total_num_cover_gt_pair[i] / total_pred_num[i] for i in range(len(total_pred_num))])

# V3
# print('\n')
# print('Finished all testing set')
# print('>>>>>>>>>>>>>>>>>>')
# print('Right predicate & in top2:', pred_right_top2)
# print('Right predicate & in others:', pred_right_others2)
# print('Right predicate & not in available classes:', pred_right_notinco)
# print('\n')
# print('Wrong predicate & in top2:', pred_wrong_top2)
# print('Wrong predicate & in others:', pred_wrong_others2)
# print('Wrong predicate & not in available classes:', pred_wrong_notinco)
# print('\n')
# print('Predicate exist based on mat:', pred_exist_based_on_mat)
# print('Predicate exist & is gt based on mat:', pred_exist_based_on_mat_ingt)
# print('Predicate not exist based on mat:', pred_not_exist_based_on_mat)
# print('Predicate not exist & but gt based on mat:', pred_not_exist_based_on_mat_butreal)
# print('<<<<<<<<<<<<<<<<<')

# V5
# print('\n')
# print('>>>>>>>>>>>>>>>>>>')
# print('total_correct number:', total_correct)
# print('total_gt_pairs number:', total_gt_pairs)
# print('ratio of correct samples in gt samples:', [total_correct[i]*1.0/total_gt_pairs[i] for i in range(len(total_correct))])
# print('total_bg_pairs number:', total_bg_pairs)
# print('total_bg_erasable_pairs number:', total_bg_erasable_pairs)
# print('ratio of erasable bgs in bgs:', [total_bg_erasable_pairs[i]*1.0/total_bg_pairs[i] for i in range(len(total_bg_pairs))])
# print('total_sample number:', total_sample)
# print('total_gt number:', total_gt)
# print('<<<<<<<<<<<<<<<<<')