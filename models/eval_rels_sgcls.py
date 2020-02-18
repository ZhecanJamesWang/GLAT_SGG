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
# from lib.kern_model import KERN
from lib.kern_model_sgcls import KERN

from lib.glat import GLATNET
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


conf = ModelConfig()

train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                          use_proposals=conf.use_proposals,
                          filter_non_overlap=conf.mode == 'sgdet')

ind_to_predicates = train.ind_to_predicates # ind_to_predicates[0] means no relationship

ind_to_classes = train.ind_to_classes

if conf.test:
    val = test

logSoftmax_0 = torch.nn.LogSoftmax(dim=0)
logSoftmax_1 = torch.nn.LogSoftmax(dim=1)

softmax_0 = torch.nn.Softmax(dim=0)
softmax_1 = torch.nn.Softmax(dim=1)

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
                return_top100=conf.return_top100)

model = GLATNET(vocab_num=[52, 153],
                feat_dim=300,
                nhid_glat_g=300,
                nhid_glat_l=300,
                nout=300,
                dropout=0.1,
                nheads=8,
                blank=152,
                types=[2]*6)

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
# ckpt_glat = torch.load('/home/tangtangwzc/KERN/saved_models/2019-11-13-00-04/with_constrant_mean__R@100.pth')
# ckpt_glat = torch.load('/home/tangtangwzc/KERN/saved_models/2019-11-13-00-04/with_constrant_mean__R@100.pth')
# ckpt_glat = torch.load('/home/tangtangwzc/KERN/saved_models/2019-11-20-17-16/with_constrant_mean__R@100.pth')
# ckpt_glat = torch.load('/home/tangtangwzc/KERN/saved_models/2019-11-20-17-14/with_constrant_mean__R@100.pth')

# ckpt_glat = torch.load('/home/tangtangwzc/KERN/saved_models/2019-11-21-00-11/with_constrant_mean__R@100.pth')
# ckpt_glat = torch.load('/home/tangtangwzc/KERN/saved_models/2019-11-21-17-04/with_constrant_mean__R@100.pth')
# ckpt_glat = torch.load('/home/tangtangwzc/KERN/saved_models/2019-11-20-17-14/with_constrant_mean__R@100.pth')
# ckpt_glat = torch.load('/home/tangtangwzc/KERN/saved_models/2019-11-21-00-00/with_constrant_mean__R@100.pth')
# ckpt_glat = torch.load('/home/tangtangwzc/KERN/saved_models/2019-11-19-17-20/with_constrant_mean__R@100.pth')
# ckpt_glat = torch.load('/home/tangtangwzc/Common_sense/models/2019-11-03-17-28_2_2_2_2_2_2_concat_no_init_mask/best_test_node_mask_predicate_acc.pth')
# ckpt_glat = torch.load('/home/tangtangwzc/KERN/saved_models/2019-11-22-15-29/with_constrant_mean__R@100.pth')

# finetuned glat(0.3)
ckpt_glat = torch.load('/home/tangtangwzc/KERN/saved_models/2019-11-20-23-59/with_constrant_mean__R@100.pth')

# ckpt_glat = torch.load('/home/haoxuan/code/KERN/checkpoints/motifnet_glat/motifnet_glat-25.tar')

optimistic_restore(model, ckpt_glat['model'])
# optimistic_restore(model, ckpt_glat['state_dict'])

print('finish pretrained loading')
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
    node_logits = []
    node_logits_dists = []

    for i in range(sample_num):
        input_class = total_data['node_class'][i]
        adj = total_data['adj'][i]
        node_type = total_data['node_type'][i]
        node_logit = total_data['node_logit'][i]
        node_logit_pad = torch.Tensor([0] * node_logit.size()[0]).unsqueeze(-1).t()
        node_logit = torch.cat((node_logit, Variable(node_logit_pad.t().cuda())), dim=1)

        node_logit_dists = total_data['node_logit_dists'][i]
        node_logit_pad_dists = torch.Tensor([0] * node_logit_dists.size()[0]).unsqueeze(-1).t()
        node_logit_dists = torch.cat((node_logit_dists, Variable(node_logit_pad_dists.t().cuda())), dim=1)

        pad_node_logit = tensor2variable(torch.zeros((max_length - input_class.size(0)), node_logit.size()[1]).cuda())
        node_logits.append(torch.cat((node_logit, pad_node_logit), 0).unsqueeze(0))

        pad_node_logit_dists = tensor2variable(torch.zeros((max_length - input_class.size(0)), node_logit_dists.size()[1]).cuda())
        node_logits_dists.append(torch.cat((node_logit_dists, pad_node_logit_dists), 0).unsqueeze(0))

        pad_input_class = tensor2variable(blank_idx * torch.ones(max_length - input_class.size(0)).long().cuda())
        input_classes.append(torch.cat((input_class, pad_input_class), 0).unsqueeze(0))
        # input_classes.append(torch.cat((input_class, blank_idx*torch.ones(max_length-input_class.size(0)).long().cuda()), 0).unsqueeze(0))

        new_adj = torch.cat((adj, torch.zeros(max_length-adj.size(0), adj.size(1)).long().cuda()), 0)
        if max_length != new_adj.size(1):
            new_adj = torch.cat((new_adj, torch.zeros(new_adj.size(0), max_length-new_adj.size(1)).long().cuda()), 1)
        adjs.append(new_adj.unsqueeze(0))
        # pdb.set_trace()
        node_types.append(
            torch.cat((node_type, 2 * torch.ones(max_length-node_type.size(0)).long().cuda()), 0).unsqueeze(0))

    input_classes = torch.cat(input_classes, 0)

    node_logits = torch.cat(node_logits, 0)
    node_logits_dists = torch.cat(node_logits_dists, 0)

    adjs = torch.cat(adjs, 0)
    adjs_lbl = adjs
    adjs_con = torch.clamp(adjs, 0, 1)
    node_types = torch.cat(node_types, 0)

    return input_classes, adjs_con, adjs_lbl, node_types, node_logits, node_logits_dists
    # return input_classes, adjs_con, adjs_lbl, node_types, node_logits


def soft_merge6(pred_entry):
    # index = (node_type == 0).squeeze(0).unsqueeze(-1).repeat(1, 52)
    logit_base_predicate = pred_entry['rel_dists']
    # .data.squeeze(0)[index].view(-1, 52)

    logit_base_predicate_50 = logit_base_predicate[:, 1:].clone()
    # logit_glat_predicate_51 = logit_glat[:, 1:].clone()

    logit_base_predicate_50 = softmax_1(logit_base_predicate_50).data
    # logit_glat_predicate_51 = softmax_1(logit_glat_predicate_51).data

    logit_base_predicate_weight = torch.max(logit_base_predicate_50, dim=1)[0]
    # logit_glat_predicate_weight = torch.max(logit_glat_predicate_51, dim=1)[0]
    cut = int(logit_base_predicate_weight.size()[0] * 0.3)

    if cut != 0:
        blank_mask = logit_base_predicate_weight.sort()[1][:cut]

        # logit_base_predicate[blank_mask] = torch.Tensor([0] * 3 + [1] + [0] * 48)
        logit_base_predicate[blank_mask].data = torch.Tensor([0] * 50 + [1]).unsqueeze(0).repeat(cut, 1)

        pred_entry['rel_dists'] = logit_base_predicate
    return pred_entry


def soft_merge4(logit_base, logit_glat, node_type):

    index = (node_type == 0).squeeze(0).unsqueeze(-1).repeat(1, 52)
    logit_base_predicate = logit_base.data.squeeze(0)[index].view(-1, 52)

    logit_base_predicate_51 = Variable(logit_base_predicate[:, 1:]).clone()
    logit_glat_predicate_51 = logit_glat[:, 1:].clone()

    logit_base_predicate_51 = softmax_1(logit_base_predicate_51).data
    logit_glat_predicate_51 = softmax_1(logit_glat_predicate_51).data

    logit_base_predicate_weight = torch.max(logit_base_predicate_51, dim=1)[0]
    logit_glat_predicate_weight = torch.max(logit_glat_predicate_51, dim=1)[0]

    output_predict_51 = torch.zeros(logit_glat_predicate_51.size()).cuda()
    mask_base_over_glat = logit_base_predicate_weight > logit_glat_predicate_weight
    mask_base_over_glat = mask_base_over_glat.unsqueeze(-1).repeat(1, 51)
    # output_predict_51[mask_base_over_glat]= logit_base_predicate_51[mask_base_over_glat]
    tmp = output_predict_51[mask_base_over_glat].view(-1, 51)
    output_predict_51[mask_base_over_glat] = output_predict_51[mask_base_over_glat].view(-1, 51)
    output_predict_51[mask_base_over_glat] = tmp
        # .view(-1, 51)

    mask_glat_over_base = logit_glat_predicate_weight > logit_base_predicate_weight
    mask_glat_over_base = mask_glat_over_base.unsqueeze(-1).repeat(1, 51)
    tmp = logit_glat_predicate_51[mask_glat_over_base].view(-1, 51)
    output_predict_51[mask_glat_over_base] = output_predict_51[mask_glat_over_base].view(-1, 51)
    output_predict_51[mask_glat_over_base] = tmp
    # .view(-1, 51)

    # mask_base_over_glat = Variable(logit_base_predicate[:, 0].clone()) > logit_glat[:, 0].clone()
    # mask_glat_over_base = logit_glat[:, 0].clone() > Variable(logit_base_predicate[:, 0].clone())
    #
    # output_predict_1 = torch.zeros(logit_glat[:, 0].size()).cuda()
    #
    # output_predict_1[mask_base_over_glat] = logit_base_predicate[:, 0][mask_base_over_glat.data]
    # output_predict_1[mask_glat_over_base] = logit_glat[:, 0].data[mask_glat_over_base.data]

    # output_logit_predicate = torch.cat((output_predict_1.unsqueeze(-1), output_predict_51,), dim=1)
    output_logit_predicate = torch.cat((logit_base_predicate[:, 0].clone().unsqueeze(-1), output_predict_51), dim=1)

    output_logit_predicate = output_logit_predicate/torch.sum(output_logit_predicate, dim=1, keepdim=True)

    return output_logit_predicate


# def soft_merge4(logit_base, logit_glat, node_type):
#
#     index = (node_type == 0).squeeze(0).unsqueeze(-1).repeat(1, 52)
#     logit_base_predicate = logit_base.data.squeeze(0)[index].view(-1, 52)
#
#     logit_base_predicate_51 = Variable(logit_base_predicate[:, 1:]).clone()
#     logit_glat_predicate_51 = logit_glat[:, 1:].clone()
#
#     logit_base_predicate_51 = softmax_1(logit_base_predicate_51).data
#     logit_glat_predicate_51 = softmax_1(logit_glat_predicate_51).data
#
#     output_predict_51 = torch.zeros(logit_glat_predicate_51.size()).cuda()
#     mask_base_over_glat = logit_base_predicate_51 > logit_glat_predicate_51
#     output_predict_51[mask_base_over_glat] = logit_base_predicate_51[mask_base_over_glat]
#     mask_glat_over_base = logit_glat_predicate_51 > logit_base_predicate_51
#     output_predict_51[mask_glat_over_base] = logit_glat_predicate_51[mask_glat_over_base]
#
#     # mask_base_over_glat = Variable(logit_base_predicate[:, 0].clone()) > logit_glat[:, 0].clone()
#     # mask_glat_over_base = logit_glat[:, 0].clone() > Variable(logit_base_predicate[:, 0].clone())
#     #
#     # output_predict_1 = torch.zeros(logit_glat[:, 0].size()).cuda()
#     #
#     # output_predict_1[mask_base_over_glat] = logit_base_predicate[:, 0][mask_base_over_glat.data]
#     # output_predict_1[mask_glat_over_base] = logit_glat[:, 0].data[mask_glat_over_base.data]
#
#     # output_logit_predicate = torch.cat((output_predict_1.unsqueeze(-1), output_predict_51,), dim=1)
#     output_logit_predicate = torch.cat((logit_base_predicate[:, 0].clone().unsqueeze(-1), output_predict_51), dim=1)
#
#     output_logit_predicate = output_logit_predicate/torch.sum(output_logit_predicate, dim=1, keepdim=True)
#
#     return output_logit_predicate


def soft_merge3(logit_base, logit_glat, node_type):

    index = (node_type == 0).squeeze(0).unsqueeze(-1).repeat(1, 52)
    logit_base_predicate = logit_base.data.squeeze(0)[index].view(-1, 52)

    logit_base_predicate = softmax_1(Variable(logit_base_predicate)).data
    logit_glat_predicate = softmax_1(logit_glat).data

    logit_base_predicate_weight = torch.max(logit_base_predicate, dim=1)[0]
    logit_glat_predicate_weight = torch.max(logit_glat_predicate, dim=1)[0]

    combined_weight = torch.cat((logit_base_predicate_weight.unsqueeze(0), logit_glat_predicate_weight.unsqueeze(0)), 0)

    combined_weight = combined_weight/torch.sum(combined_weight, dim=0, keepdim=True)

    logit_base_predicate_weight = combined_weight[0,:].unsqueeze(-1).repeat(1, 52)
    logit_glat_predicate_weight = combined_weight[1,:].unsqueeze(-1).repeat(1, 52)

    logit_base_predicate = logit_base_predicate * logit_base_predicate_weight
    logit_glat = logit_glat.data * logit_glat_predicate_weight

    output_logit_predicate = logit_base_predicate + logit_glat

    # output_logit_predicate = output_logit_predicate/torch.sum(output_logit_predicate, dim=1, keepdim=True)
    output_logit_predicate = softmax_1(Variable(output_logit_predicate))

    return output_logit_predicate


def soft_merge2(logit_base, logit_glat, node_type):

    index = (node_type == 0).squeeze(0).unsqueeze(-1).repeat(1, 52)
    logit_base_predicate = logit_base.data.squeeze(0)[index].view(-1, 52)

    logit_base_predicate_51 = Variable(logit_base_predicate[:, 1:]).clone()
    logit_glat_predicate_51 = logit_glat[:, 1:].clone()
    logit_base_predicate_51 = softmax_1(logit_base_predicate_51).data
    logit_glat_predicate_51 = softmax_1(logit_glat_predicate_51).data

    logit_base_predicate_weight = torch.max(logit_base_predicate_51, dim=1)[0]
    logit_glat_predicate_weight = torch.max(logit_glat_predicate_51, dim=1)[0]

    combined_weight = torch.cat((logit_base_predicate_weight.unsqueeze(0), logit_glat_predicate_weight.unsqueeze(0)), 0)

    combined_weight = combined_weight/torch.sum(combined_weight, dim=0, keepdim=True)

    logit_base_predicate_weight = combined_weight[0,:].unsqueeze(-1).repeat(1, 52)
    logit_glat_predicate_weight = combined_weight[1,:].unsqueeze(-1).repeat(1, 52)

    logit_base_predicate = logit_base_predicate * logit_base_predicate_weight
    logit_glat = logit_glat.data * logit_glat_predicate_weight

    output_logit_predicate = logit_base_predicate + logit_glat

    # output_logit_predicate = output_logit_predicate/torch.sum(output_logit_predicate, dim=1, keepdim=True)
    output_logit_predicate = softmax_1(Variable(output_logit_predicate))

    return output_logit_predicate


def soft_merge1(logit_base, logit_glat, node_type):

    index = (node_type == 0).squeeze(0).unsqueeze(-1).repeat(1, 52)
    logit_base_predicate = logit_base.data.squeeze(0)[index].view(-1, 52)

    logit_base_predicate = logSoftmax_1(Variable(logit_base_predicate))
    logit_glat = logSoftmax_1(logit_glat)

    logit_base_predicate_weight = torch.max(logit_base_predicate[:, 1:], dim=1)[0]
    logit_glat_predicate_weight = torch.max(logit_glat[:, 1:], dim=1)[0]

    combined_weight = torch.cat((logit_base_predicate_weight.unsqueeze(0),
                                 Variable(logit_glat_predicate_weight.data.unsqueeze(0))), 0)

    combined_weight = softmax_0(combined_weight)

    logit_base_predicate_weight = combined_weight[0,:].unsqueeze(-1).repeat(1, 52)
    logit_glat_predicate_weight = combined_weight[1,:].unsqueeze(-1).repeat(1, 52)

    # ones = torch.ones(logit_base_predicate_weight.size()[0], 1)

    # logit_base_predicate_weight = torch.cat((ones.cuda(), logit_base_predicate_weight.data), 1)
    # logit_glat_predicate_weight = torch.cat((ones.cuda(), logit_glat_predicate_weight.data), 1)

    logit_base_predicate = logit_base_predicate * logit_base_predicate_weight
    logit_glat = logit_glat.data * logit_glat_predicate_weight.data

    output_logit_predicate = logit_base_predicate.data + logit_glat

    # output_logit_51 = Variable(logit_base_predicate[:, 1:]).clone()

    output_logit_predicate = softmax_1(Variable(output_logit_predicate)).data

    return output_logit_predicate


def glat_wrapper(total_data):
    # Batch size assumed to be 1
    # input_class, adjs_con, adjs_lbl, node_type, node_logit = my_collate(total_data)
    input_class, adjs_con, adjs_lbl, node_type, node_logit, node_logit_dists = my_collate(total_data)
    if torch.is_tensor(input_class):
        input_class = Variable(input_class)
    if torch.is_tensor(node_logit):
        node_logit = Variable(node_logit)
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

    pred_label_predicate = soft_merge4(node_logit_dists, pred_label_predicate, node_type)

    return pred_label_predicate, pred_label_entities
    # return pred_label_predicate.data.cpu().numpy(), pred_label_entities.data.cpu().numpy()


def glat_postprocess(pred_entry, if_predicting=False):
    # pred_entry = {
    #     'pred_boxes': boxes_i * BOX_SCALE / IM_SCALE,  # (23, 4) (16, 4)
    #     'pred_classes': objs_i,  # (23,) (16,)
    #     'pred_rel_inds': rels_i,  # (506, 2) (240, 2)
    #     'obj_scores': obj_scores_i,  # (23,) (16,)
    #     'rel_scores': pred_scores_i,  # hack for now. (506, 51) (240, 51)
    # }

    if if_predicting:
        pred_entry = numpy2cuda_dict(pred_entry)

    pred_entry['rel_dists'] = tensor2variable(pred_entry['rel_dists'])

    #sgcls
    pred_entry['ent_dists'] = tensor2variable(pred_entry['ent_dists'])

    pred_entry['rel_scores'] = tensor2variable(pred_entry['rel_scores'])
    pred_entry['pred_classes'] = tensor2variable(pred_entry['pred_classes'])

    # pred_entry['rel_classes'] = torch.max(pred_entry['rel_scores'][:, 1:], dim=1)[1].unsqueeze(1) + 1
    pred_entry['rel_classes'] = torch.max(pred_entry['rel_dists'][:, 1:], dim=1)[1].unsqueeze(1) + 1

    pred_entry['rel_classes'] = variable2tensor(pred_entry['rel_classes'])
    pred_entry['pred_relations'] = torch.cat((pred_entry['pred_rel_inds'], pred_entry['rel_classes']), dim=1)

    # pred_entry = soft_merge6(pred_entry)

    # total_data = build_graph_structure(pred_entry, ind_to_classes, ind_to_predicates, if_predicting=if_predicting)

    # For SGCLS
    if conf.mode == "sgcls" or conf.mode == "sgdet":
        total_data, useless_entity_id = build_graph_structure(pred_entry, ind_to_classes, ind_to_predicates,
                                                              if_predicting=if_predicting,
                                                              sgclsdet=True)
    else:
        total_data = build_graph_structure(pred_entry, ind_to_classes, ind_to_predicates, if_predicting=if_predicting)

    pred_label_predicate, pred_label_entities = glat_wrapper(total_data)

    # pred_entry['rel_scores'] = pred_label_predicate

    # For SGCLS
    pred_entry['rel_scores'] = pred_label_predicate[:, :-1]

    # For SGCLS
    pred_entry['entity_scores'] = pred_label_entities[:, :-2]
    pred_entry['pred_classes'] = pred_label_entities.max(1)[1]

    # return pred_entry
    if conf.mode == "sgcls" or conf.mode == "sgdet":
        return pred_entry, useless_entity_id
    else:
        return pred_entry


all_pred_entries = []


def val_batch(batch_num, b, evaluator, evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list,
              thrs=(20, 50, 100)):
    det_res = detector[b]

    if conf.num_gpus == 1:
        det_res = [det_res]

    for i, det in enumerate(det_res):

        det = det[1]

        if len(det) == 6 and not conf.return_top100:
            # (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i, rel_dists) = det

            # sgcls
            (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i, rel_dists, ent_dists) = det
            rels_i_a100 = np.asarray([])
        else:
            # (boxes_i, objs_i, obj_scores_i, rels_i_b100, pred_scores_i_b100, rels_i_a100, pred_scores_i_a100,
            #  rel_scores_idx_b100, rel_scores_idx_a100, rel_dists) = det

            # sgcls
            (boxes_i, objs_i, obj_scores_i, rels_i_b100, pred_scores_i_b100, rels_i_a100, pred_scores_i_a100,
             rel_scores_idx_b100, rel_scores_idx_a100, rel_b_dists, rel_a_dists, ent_dists) = det
            rel_dists = rel_b_dists

        # print("boxes_i.size(): ", boxes_i.shape)
        # print("rels_i_b100.size(): ", rels_i_b100.shape)
        # print("rels_i_a100.size(): ", rels_i_a100.shape)

        gt_entry = {
            'gt_classes': val.gt_classes[batch_num + i].copy(), #(23,) (16,)
            'gt_relations': val.relationships[batch_num + i].copy(), #(29, 3) (6, 3)
            'gt_boxes': val.gt_boxes[batch_num + i].copy(), #(23, 4) (16, 4)
        }

        if conf.return_top100:
            pred_entry = {
                'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE, #(23, 4) (16, 4)
                'pred_classes': objs_i, #(23,) (16,)
                'pred_rel_inds': rels_i_b100, #(506, 2) (240, 2)
                'obj_scores': obj_scores_i, #(23,) (16,)
                'rel_scores': pred_scores_i_b100,  # hack for now. (506, 51) (240, 51)
                'rel_dists': rel_dists.data.cpu().numpy(),
                'ent_dists': ent_dists.data.cpu().numpy()
            }
        else:
            pred_entry = {
                'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE, #(23, 4) (16, 4)
                'pred_classes': objs_i, #(23,) (16,)
                'pred_rel_inds': rels_i, #(506, 2) (240, 2)
                'obj_scores': obj_scores_i, #(23,) (16,)
                'rel_scores': pred_scores_i,  # hack for now. (506, 51) (240, 51)
                'rel_dists': rel_dists.data.cpu().numpy(),
                'ent_dists': ent_dists.data.cpu().numpy()
            }

        pred_entry = glat_postprocess(pred_entry, if_predicting=True)

        # For SGCLS
        if conf.mode == "sgcls" or conf.mode == "sgdet":
            useless_entity_id = pred_entry[1]
            pred_entry = pred_entry[0]

        pred_entry = cuda2numpy_dict(pred_entry)

        # without adding a_100

        # pred_entry['rel_scores'] = pred_entry['rel_scores'][:, :-1]

        # conditioning on adding a_100

        # if len(rels_i_a100.shape) == 1:
        #     # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #     pred_entry['rel_scores'] = pred_entry['rel_scores'][:, :-1]
        #     # pred_entry['rel_scores'] = rel_scores_one_hot
        #     # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # else:
        #
        #     pred_entry['pred_rel_inds'] = np.concatenate((pred_entry['pred_rel_inds'], rels_i_a100), axis=0)
        # # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #     pred_entry['rel_scores'] = np.concatenate((pred_entry['rel_scores'][:, :-1], pred_scores_i_a100), axis=0)
        #     # pred_entry['rel_scores'] = np.concatenate((rel_scores_one_hot, pred_scores_i_a100), axis=0)
        # # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # pred_entry = rank_predicate(pred_entry)


    # for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):
    #     gt_entry = {
    #         'gt_classes': val.gt_classes[batch_num + i].copy(),
    #         'gt_relations': val.relationships[batch_num + i].copy(),
    #         'gt_boxes': val.gt_boxes[batch_num + i].copy(),
    #     }
    #     assert np.all(objs_i[rels_i[:,0]] > 0) and np.all(objs_i[rels_i[:,1]] > 0)
    #     # assert np.all(rels_i[:,2] > 0)
    #
    #     pred_entry = {
    #         'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
    #         'pred_classes': objs_i,
    #         'pred_rel_inds': rels_i,
    #         'obj_scores': obj_scores_i,
    #         'rel_scores': pred_scores_i,
    #     }

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

for val_b, batch in enumerate(tqdm(val_loader)):
    val_batch(conf.num_gpus * val_b, batch, evaluator, evaluator_multiple_preds, evaluator_list,
              evaluator_multiple_preds_list)
    torch.cuda.empty_cache()

recall = evaluator[conf.mode].print_stats()
recall_mp = evaluator_multiple_preds[conf.mode].print_stats()

mean_recall = calculate_mR_from_evaluator_list(evaluator_list, conf.mode, save_file=conf.save_rel_recall)
mean_recall_mp = calculate_mR_from_evaluator_list(evaluator_multiple_preds_list, conf.mode, multiple_preds=True,
                                                  save_file=conf.save_rel_recall)

torch.cuda.empty_cache()

torch.cuda.empty_cache()

if conf.cache is not None:
    with open(conf.cache, 'wb') as f:
        pkl.dump(all_pred_entries, f)
