
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

# Adding vis fea
# from lib.glat import GLATNET
from lib.glat_fea import GLATNET

from torch.autograd import Variable
import pdb
from torch.nn import functional as F
import copy
import math

conf = ModelConfig()

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
    from lib.stanford_model import RelModelStanford as RelModel
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
    from lib.motifnet_model import RelModel


if conf.model_s_m == 'motifnet' or conf.model_s_m == 'stanford':
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
elif conf.model_s_m =='kern':
    from lib.kern_model import KERN
    detector = KERN(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                    num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                    use_resnet=conf.use_resnet, use_proposals=conf.use_proposals,
                    use_ggnn_obj=conf.use_ggnn_obj, ggnn_obj_time_step_num=conf.ggnn_obj_time_step_num,
                    ggnn_obj_hidden_dim=conf.ggnn_obj_hidden_dim, ggnn_obj_output_dim=conf.ggnn_obj_output_dim,
                    use_obj_knowledge=conf.use_obj_knowledge, obj_knowledge=conf.obj_knowledge,
                    use_ggnn_rel=conf.use_ggnn_rel, ggnn_rel_time_step_num=conf.ggnn_rel_time_step_num,
                    ggnn_rel_hidden_dim=conf.ggnn_rel_hidden_dim, ggnn_rel_output_dim=conf.ggnn_rel_output_dim,
                    use_rel_knowledge=conf.use_rel_knowledge, rel_knowledge=conf.rel_knowledge,
                    return_top100=True, return_unbias_logit=True, return_vis_fea=True)
else:
    print('wrong model name')

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
    print('loading pretrained model')
    ckpt_glat = torch.load(
        '/home/tangtangwzc/Common_sense/models/2019-12-18-16-08_2_2_2_2_2_2_concat_no_init_mask/best_test_node_mask_predicate_acc.pth')

    # ckpt_glat = torch.load(
    #     '/home/tangtangwzc/Common_sense/models/2019-11-03-17-28_2_2_2_2_2_2_concat_no_init_mask/best_test_node_mask_predicate_acc.pth')

elif conf.model_s_m == 'motifnet':
    # Finetuned model v0
    # print('loading Finetuned model v0')
    # ckpt_glat = torch.load('/home/haoxuan/code/KERN/checkpoints/motifnet_glat_predcls_mask_logit_thres_mbz/motifnet_glat-8.tar')

    # Finetuned model v1
    # print('loading Finetuned model v1')
    # ckpt_glat = torch.load('/home/haoxuan/code/KERN/checkpoints/motifnet_glat_mask_pred/motifnet_glat-49.tar')

    # Finetuned model v2
    # print('loading Finetuned model v2')
    # ckpt_glat = torch.load('/home/haoxuan/code/KERN/checkpoints/motifnet_glat_predcls_mask_logit_thres_train_v2mbz/motifnet_glat-37.tar')

    # Finetuned model v3
    # print('loading Finetuned model v3')
    # ckpt_glat = torch.load('/home/haoxuan/code/KERN/checkpoints/motifnet_glat_sgcls_mask_logit_thres_train_v3/motifnet_glat-26.tar')

    # print('loading Finetuned model v3_1')
    # ckpt_glat = torch.load('/home/haoxuan/code/KERN/checkpoints/motifnet_glat_sgcls_mask_logit_thres_train_v3_1/motifnet_glat-48.tar')

    print('loading motif vis sgcls model')
    ckpt_glat = torch.load('/home/haoxuan/code/KERN/checkpoints/motif_glat_visfea_train/motifnet_glat-48.tar')


    # Pretrained Model
    # print('loading pretrained model')
    # ckpt_glat = torch.load(
    #     '/home/tangtangwzc/Common_sense/models/2019-12-18-16-08_2_2_2_2_2_2_concat_no_init_mask/best_test_node_mask_predicate_acc.pth')

    # ckpt_glat = torch.load(
    #     '/home/tangtangwzc/Common_sense/models/2019-11-03-17-28_2_2_2_2_2_2_concat_no_init_mask/best_test_node_mask_predicate_acc.pth')

elif conf.model_s_m == 'kern':
    print('loading kern vis sgcls model')
    ckpt_glat = torch.load('/home/haoxuan/code/KERN/checkpoints/kern_glat_visfea_train/motifnet_glat-49.tar')


# Finetuned model
optimistic_restore(model, ckpt_glat['state_dict'])

# Pretrained Model
# optimistic_restore(model, ckpt_glat['model'])
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

    # Adding vis fea >>>>>>>>>>
    entity_visfea = total_data['entity_visfea']
    if torch.is_tensor(total_data['entity_visfea']):
        entity_visfea = Variable(total_data['entity_visfea'])
    rel_visfea = total_data['rel_visfea']
    if torch.is_tensor(total_data['rel_visfea']):
        rel_visfea = Variable(total_data['rel_visfea'])
    # <<<<<<<<<<<<<<<<<<

    # Adding vis fea >>>>>>>>>>
    pred_label, pred_connect = model(input_class, adj_con, node_type, entity_visfea, rel_visfea)
    # <<<<<<<<<<<<<<<<<<


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

    # adding vis fea >>>>>>>
    pred_entry['entity_visfea'] = tensor2variable(pred_entry['entity_visfea'])
    pred_entry['rel_visfea'] = tensor2variable(pred_entry['rel_visfea'])
    assert pred_entry['rel_visfea'].size(0) == pred_entry['rel_scores'].size(0)
    assert pred_entry['entity_visfea'].size(0) == pred_entry['pred_classes'].size(0)
    # <<<<<<<


    pred_entry['rel_classes'] = torch.max(pred_entry['rel_scores'][:, 1:], dim=1)[1].unsqueeze(1) + 1
    if mask_idx is not None:
        pred_entry['rel_classes'][mask_idx] = 51
    pred_entry['rel_classes'] = variable2tensor(pred_entry['rel_classes'])
    # pdb.set_trace()
    pred_entry['pred_relations'] = torch.cat((pred_entry['pred_rel_inds'], pred_entry['rel_classes']), dim=1)

    if conf.mode == "sgcls" or conf.mode == "sgdet":
        total_data, useless_entity_id = build_graph_structure(pred_entry, ind_to_classes, ind_to_predicates, if_predicting=if_predicting,
                                           sgclsdet=True)
    else:
        total_data = build_graph_structure(pred_entry, ind_to_classes, ind_to_predicates, if_predicting=if_predicting)

    pred_label_predicate, pred_label_entities = glat_wrapper(total_data)

    pred_entry['rel_scores'] = pred_label_predicate
    # pdb.set_trace()
    # For SGCLS

    # pdb.set_trace()

    # softmax
    # pred_entry['obj_scores_rm'] = pred_label_entities
    # pred_entry['obj_scores'] = F.softmax(pred_label_entities, dim=1).max(1)[0]

    # For bug0
    if conf.mode == "sgcls" or conf.mode == "sgdet":
        pred_entry['obj_scores_rm'] = pred_label_entities
        pred_entry['obj_scores'] = F.softmax(pred_label_entities, dim=1).max(1)[0]
        pred_entry['pred_classes'] = pred_label_entities.max(1)[1]

        return pred_entry, useless_entity_id
    else:
        return pred_entry



all_pred_entries = []

def val_batch(batch_num, b, evaluator,evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list, accs):
    # det_res = detector[b]
    # dict_gt, det_res = detector[b]
    vis_result, bias_logit, det_res = detector[b]
    # Adding vis fea
    vis_result[1] = vis_result[1][det_res[-2]]
    obj_visfea, rel_visfea = vis_result

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
                # adding vis fea >>>>>>>
                'entity_visfea': obj_visfea,  # (num_entities, 4096) Tensor Variable
                'rel_visfea': rel_visfea, # (num_predicate, 4096) Tensor Variable
                # <<<<<<<<<<<<<<<<<
            }

        else:
            pred_entry = {
                'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE, #(23, 4) (16, 4)
                'pred_classes': objs_i, #(23,) (16,)
                'pred_rel_inds': rels_i, #(506, 2) (240, 2)
                'obj_scores': obj_scores_i, #(23,) (16,)
                'rel_scores': pred_scores_i,  # hack for now. (506, 51) (240, 51)
            }

        pred_entry = glat_postprocess(pred_entry, if_predicting=True, mask_idx=None)


        # For SGCLS
        if conf.mode == "sgcls" or conf.mode == "sgdet":
            useless_entity_id = pred_entry[1]
            pred_entry = pred_entry[0]

        pred_entry = cuda2numpy_dict(pred_entry)

        if len(rels_i_a100.shape) == 1:
            pred_entry['rel_scores'] = pred_entry['rel_scores'][:, :-1]
        else:
            pred_entry['pred_rel_inds'] = np.concatenate((pred_entry['pred_rel_inds'], rels_i_a100), axis=0)
            pred_entry['rel_scores'] = np.concatenate((pred_entry['rel_scores'][:, :-1], pred_scores_i_a100), axis=0)

        all_pred_entries.append(pred_entry)

        eval_entry(conf.mode, gt_entry, pred_entry, evaluator, evaluator_multiple_preds,
                   evaluator_list, evaluator_multiple_preds_list)



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
accs = [0, 0]
# number_mask = [0]

# threshold = conf.threshold
# print('Threshold is', threshold)
# rel_matrix = np.load('./prior_matrices/rel_matrix.npy')
for val_b, batch in enumerate(tqdm(val_loader)):
    # val_batch(conf.num_gpus*val_b, batch, evaluator)
    val_batch(conf.num_gpus * val_b, batch, evaluator, evaluator_multiple_preds, evaluator_list,
              evaluator_multiple_preds_list, accs)

    torch.cuda.empty_cache()

# print('test acc of mask:', accs[0] * 1.0 / accs[1])
# print('number of mask:', accs[1])
# print('total number of mask:', number_mask)

recall = evaluator[conf.mode].print_stats()
recall_mp = evaluator_multiple_preds[conf.mode].print_stats()

mean_recall = calculate_mR_from_evaluator_list(evaluator_list, conf.mode, save_file=conf.save_rel_recall)
mean_recall_mp = calculate_mR_from_evaluator_list(evaluator_multiple_preds_list, conf.mode, multiple_preds=True,
                                                  save_file=conf.save_rel_recall)
# evaluator[conf.mode].print_stats()

if conf.cache is not None:
    with open(conf.cache,'wb') as f:
        pkl.dump(all_pred_entries, f)