
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
# from lib.glat_logit import GLATNET
from lib.glat import GLATNET
from torch.autograd import Variable
import pdb

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


conf = ModelConfig()
if conf.model_s_m == 'motifnet':
    from lib.motifnet_model_working import RelModel
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

logSoftmax_0 = torch.nn.LogSoftmax(dim=0)
logSoftmax_1 = torch.nn.LogSoftmax(dim=1)

softmax_0 = torch.nn.Softmax(dim=0)
softmax_1 = torch.nn.Softmax(dim=1)

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
                types=[2] * 6)

detector.cuda()
ckpt = torch.load(conf.ckpt)
detector.eval()

optimistic_restore(detector, ckpt['state_dict'])
# if conf.mode == 'sgdet':
#     det_ckpt = torch.load('checkpoints/new_vgdet/vg-19.tar')['state_dict']
#     detector.detector.bbox_fc.weight.data.copy_(det_ckpt['bbox_fc.weight'])
#     detector.detector.bbox_fc.bias.data.copy_(det_ckpt['bbox_fc.bias'])
#     detector.detector.score_fc.weight.data.copy_(det_ckpt['score_fc.weight'])
#     detector.detector.score_fc.bias.data.copy_(det_ckpt['score_fc.bias'])

if conf.model_s_m == 'stanford':
    # ckpt_glat = torch.load('/home/haoxuan/code/KERN/checkpoints/motifnet_glat/motifnet_glat-25.tar')
    # ckpt_glat = torch.load('/home/haoxuan/code/KERN/checkpoints/stanford_glat_1/stanford_glat-20.tar')

    # stanford predcls finetune glat
    ckpt_glat = torch.load('/home/haoxuan/code/KERN/checkpoints/stanford_glat_predcls/stanford_glat-21.tar')

elif conf.model_s_m == 'motifnet':
    # ckpt_glat = torch.load('/home/haoxuan/code/KERN/checkpoints/motifnet_glat/motifnet_glat-25.tar')
    # ckpt_glat = torch.load('/home/haoxuan/code/KERN/checkpoints/stanford_glat_1/stanford_glat-20.tar')

    # # motif predcls finetune glat
    # ckpt_glat = torch.load('/home/haoxuan/code/KERN/checkpoints/motifnet_glat_predcls_mbz/motifnet_glat-27.tar')

    # # self finetune predcls motif glat weight
    # ckpt_glat=torch.load('/home/tangtangwzc/KERN/checkpoints/motifnet_glat_predcls_mbz_v2_2020_0202_2121/motifnet_glat-9.tar')

    # # # self finetune predcls motif glat LOGIT weight
    # ckpt_glat = torch.load('/home/tangtangwzc/KERN/checkpoints/motifnet_glat_predcls_mbz_v2_2020_0204_1738//motifnet_glat-20.tar')

    # # # self finetune sgcls motif glat weight
    ckpt_glat = torch.load('/home/tangtangwzc/KERN/checkpoints/motifnet_glat_sgcls_2020_0211_2317/motifnet_glat-20.tar')

# # ---------------pretrained model mask ratio 0.5
# ckpt_glat = torch.load('/home/tangtangwzc/Common_sense/models/2019-11-03-17-51_2_2_2_2_2_2_concat_no_init_mask/best_test_node_mask_predicate_acc.pth')

# # ---------------pretrained model mask ratio 0.3
# ckpt_glat = torch.load('/home/tangtangwzc/Common_sense/models/2019-12-18-16-08_2_2_2_2_2_2_concat_no_init_mask/best_test_node_mask_predicate_acc.pth')

# ckpt_glat = torch.load('/home/tangtangwzc/KERN/checkpoints/stanford_glat_1216_1/stanford_glat-6.tar')

# ckpt_glat = torch.load('/home/tangtangwzc/KERN/checkpoints/stanford_glat_1218_3/stanford_glat-10.tar')

# ckpt_glat = torch.load('/home/tangtangwzc/KERN/checkpoints/stanford_glat_1220_6/stanford_glat-9.tar')

# ckpt_glat = torch.load('/home/tangtangwzc/KERN/checkpoints/stanford_glat_1222_0110/stanford_glat-11.tar')

# ckpt_glat = torch.load('/home/tangtangwzc/KERN/checkpoints/stanford_glat_1220_6/stanford_glat-17.tar')

optimistic_restore(model, ckpt_glat['state_dict'])
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

        # pad_node_logit = tensor2variable(torch.zeros((max_length - input_class.size(0)), node_logit.size()[1]).cuda())
        # node_logits.append(torch.cat((node_logit, pad_node_logit), 0).unsqueeze(0))
        #
        # pad_node_logit_dists = tensor2variable(torch.zeros((max_length - input_class.size(0)), node_logit_dists.size()[1]).cuda())
        # node_logits_dists.append(torch.cat((node_logit_dists, pad_node_logit_dists), 0).unsqueeze(0))

        if max_length - input_class.size(0) != 0:
            pad_node_logit = tensor2variable(torch.zeros((max_length - input_class.size(0)), node_logit.size()[1]).cuda())
            node_logit = torch.cat((node_logit, pad_node_logit), 0).unsqueeze(0)
        else:
            node_logit = node_logit.unsqueeze(0)

        node_logits.append(node_logit)

        if max_length - input_class.size(0) != 0:
                pad_node_logit_dists = tensor2variable(torch.zeros((max_length - input_class.size(0)), node_logit_dists.size()[1]).cuda())
                node_logit_dists = torch.cat((node_logit_dists, pad_node_logit_dists), 0).unsqueeze(0)
        else:
            node_logit_dists = node_logit_dists.unsqueeze(0)

        node_logits_dists.append(node_logit_dists)

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

    node_logits = torch.cat(node_logits, 0)
    node_logits_dists = torch.cat(node_logits_dists, 0)

    adjs = torch.cat(adjs, 0)
    adjs_lbl = adjs
    adjs_con = torch.clamp(adjs, 0, 1)
    node_types = torch.cat(node_types, 0)

    return input_classes, adjs_con, adjs_lbl, node_types, node_logits, node_logits_dists
    # return input_classes, adjs_con, adjs_lbl, node_types


def soft_merge7(logit_base, logit_glat, node_type):

    index = (node_type == 0).squeeze(0).unsqueeze(-1).repeat(1, 52)
    logit_base_predicate = logit_base.data.squeeze(0)[index].view(-1, 52)

    logit_base_predicate_52 = Variable(logit_base_predicate).clone()
    logit_glat_predicate_52 = logit_glat.clone()

    logit_base_predicate_52 = softmax_1(logit_base_predicate_52).data
    logit_glat_predicate_52 = softmax_1(logit_glat_predicate_52).data

    logit_base_predicate_weight = torch.max(logit_base_predicate_52, dim=1)[0]
    logit_glat_predicate_weight = torch.max(logit_glat_predicate_52, dim=1)[0]

    output_predict_52 = torch.zeros(logit_glat_predicate_52.size()).cuda()
    mask_base_over_glat = logit_base_predicate_weight > logit_glat_predicate_weight
    mask_base_over_glat = mask_base_over_glat.unsqueeze(-1).repeat(1, 52)
    # output_predict_51[mask_base_over_glat]= logit_base_predicate_51[mask_base_over_glat]
    tmp = output_predict_52[mask_base_over_glat].view(-1, 52)
    output_predict_52[mask_base_over_glat] = output_predict_52[mask_base_over_glat].view(-1, 52)
    output_predict_52[mask_base_over_glat] = tmp
        # .view(-1, 51)

    mask_glat_over_base = logit_glat_predicate_weight > logit_base_predicate_weight
    mask_glat_over_base = mask_glat_over_base.unsqueeze(-1).repeat(1, 52)
    tmp = logit_glat_predicate_52[mask_glat_over_base].view(-1, 52)
    output_predict_52[mask_glat_over_base] = output_predict_52[mask_glat_over_base].view(-1, 52)
    output_predict_52[mask_glat_over_base] = tmp
    # .view(-1, 51)

    # mask_base_over_glat = Variable(logit_base_predicate[:, 0].clone()) > logit_glat[:, 0].clone()
    # mask_glat_over_base = logit_glat[:, 0].clone() > Variable(logit_base_predicate[:, 0].clone())
    #
    # output_predict_1 = torch.zeros(logit_glat[:, 0].size()).cuda()
    #
    # output_predict_1[mask_base_over_glat] = logit_base_predicate[:, 0][mask_base_over_glat.data]
    # output_predict_1[mask_glat_over_base] = logit_glat[:, 0].data[mask_glat_over_base.data]

    # output_logit_predicate = torch.cat((output_predict_1.unsqueeze(-1), output_predict_51,), dim=1)
    # output_logit_predicate = torch.cat((logit_base_predicate[:, 0].clone().unsqueeze(-1), output_predict_51), dim=1)

    # output_logit_predicate = output_logit_predicate/torch.sum(output_logit_predicate, dim=1, keepdim=True)

    output_logit_predicate = output_predict_52

    return output_logit_predicate


def soft_merge6(pred_entry):
    # index = (node_type == 0).squeeze(0).unsqueeze(-1).repeat(1, 52)
    logit_base_predicate = pred_entry['rel_dists']
    # .data.squeeze(0)[index].view(-1, 52)

    logit_base_predicate_50 = logit_base_predicate[:, 1:].clone()
    # logit_glat_predicate_51 = logit_glat[:, 1:].clone()

    logit_base_predicate_50 = softmax_1(logit_base_predicate_50).data
    # logit_glat_predicate_51 = softmax_1(logit_glat_predicate_51).data

    threshold = 0

    logit_base_predicate_weight = torch.max(logit_base_predicate_50, dim=1)[0]
    # logit_glat_predicate_weight = torch.max(logit_glat_predicate_51, dim=1)[0]
    # cut = int(logit_base_predicate_weight.size()[0] * 0.3)

    # blank_mask = (logit_base_predicate_weight < threshold).unsqueeze(-1).repeat(1, 51)
    # non_blank_mask = (logit_base_predicate_weight >= threshold).unsqueeze(-1).repeat(1, 51)

    blank_mask = ((logit_base_predicate_weight < threshold) == 1).nonzero()
    # non_blank_mask = ((logit_base_predicate_weight >= threshold) == 1).nonzero()


    # output_logit_base_predicate = torch.zeros(logit_base_predicate.size()).cuda()

    # logit_base_predicate[blank_mask] = logit_base_predicate[blank_mask].view(-1, 51)
    # size = blank_mask.size()[0]
    # output_logit_base_predicate[blank_mask].data = torch.Tensor([0] * 50 + [1]).unsqueeze(0).repeat(size, 1)

    # logit_base_predicate[blank_mask.squeeze(1), :].data = torch.Tensor([0] * 50 + [1]).unsqueeze(0).repeat(size, 1)
    # logit_base_predicate[blank_mask.squeeze(1), :].data = torch.Tensor([0] * 50 + [1]).unsqueeze(0)

    for index in blank_mask:
        logit_base_predicate[index, :].data = torch.Tensor([0] * 50 + [1]).unsqueeze(0)

    # output_logit_base_predicate[non_blank_mask].data = logit_base_predicate[non_blank_mask]

    # mask_glat_over_base = mask_glat_over_base.unsqueeze(-1).repeat(1, 51)
    # tmp = logit_glat_predicate_51[mask_glat_over_base].view(-1, 51)
    # output_predict_51[mask_glat_over_base] = output_predict_51[mask_glat_over_base].view(-1, 51)
    # output_predict_51[mask_glat_over_base] = tmp
    #
    # if cut != 0:
    #     blank_mask = logit_base_predicate_weight.sort()[1][:cut]
    #
    #     # logit_base_predicate[blank_mask] = torch.Tensor([0] * 3 + [1] + [0] * 48)
    #     logit_base_predicate[blank_mask].data = torch.Tensor([0] * 50 + [1]).unsqueeze(0).repeat(cut, 1)
    #
    pred_entry['rel_dists'] = logit_base_predicate
    return pred_entry


def soft_merge6_original(pred_entry):
    # index = (node_type == 0).squeeze(0).unsqueeze(-1).repeat(1, 52)
    logit_base_predicate = pred_entry['rel_dists']

    # .data.squeeze(0)[index].view(-1, 52)

    logit_base_predicate_50 = logit_base_predicate[:, 1:].clone()
    # logit_glat_predicate_51 = logit_glat[:, 1:].clone()

    logit_base_predicate_50 = softmax_1(logit_base_predicate_50).data
    # logit_glat_predicate_51 = softmax_1(logit_glat_predicate_51).data

    logit_base_predicate_weight = torch.max(logit_base_predicate_50, dim=1)[0]
    # logit_glat_predicate_weight = torch.max(logit_glat_predicate_51, dim=1)[0]

    threshold = 0.3
    cut = int(logit_base_predicate_weight.size()[0] * threshold)

    if cut != 0:
        blank_mask = logit_base_predicate_weight.sort()[1][:cut]

        # logit_base_predicate[blank_mask] = torch.Tensor([0] * 3 + [1] + [0] * 48)
        # logit_base_predicate[blank_mask].data = torch.Tensor([0] * 50 + [1]).unsqueeze(0).repeat(cut, 1)

        # for index in blank_mask:
        #     logit_base_predicate[index, :] = torch.Tensor([0] * 50 + [1]).unsqueeze(0)
        output = torch.Tensor()
        for index in range(logit_base_predicate.size()[0]):
            if index in blank_mask:
                if len(output.size()) == 0:
                    output = Variable(torch.Tensor([0] * 50 + [1]).unsqueeze(0)).cuda()
                else:
                    output = torch.cat((output, Variable(torch.Tensor([0] * 50 + [1]).unsqueeze(0)).cuda()), dim=0)
            else:
                if len(output.size()) == 0:
                    output = logit_base_predicate[index, :].unsqueeze(0)
                else:
                    output = torch.cat((output, logit_base_predicate[index, :].unsqueeze(0)), dim=0)

        output = (torch.max(output[:, 1:], dim=1)[1].unsqueeze(1) + 1)
        output = torch.cat((pred_entry['pred_relations'][:,:2].long(), output.data), dim=1)

        # pred_entry['rel_dists'] = output
        # pred_entry['rel_dists'] = logit_base_predicate
        pred_entry['pred_relations'] = output

    return pred_entry


# def soft_merge6_original(pred_entry):
#     # index = (node_type == 0).squeeze(0).unsqueeze(-1).repeat(1, 52)
#     logit_base_predicate = pred_entry['rel_dists']
#     # .data.squeeze(0)[index].view(-1, 52)
#
#     logit_base_predicate_50 = logit_base_predicate[:, 1:].clone()
#     # logit_glat_predicate_51 = logit_glat[:, 1:].clone()
#
#     logit_base_predicate_50 = softmax_1(logit_base_predicate_50).data
#     # logit_glat_predicate_51 = softmax_1(logit_glat_predicate_51).data
#
#     logit_base_predicate_weight = torch.max(logit_base_predicate_50, dim=1)[0]
#     # logit_glat_predicate_weight = torch.max(logit_glat_predicate_51, dim=1)[0]
#
#     cut = int(logit_base_predicate_weight.size()[0] * 0.3)
#
#     if cut != 0:
#         blank_mask = logit_base_predicate_weight.sort()[1][:cut]
#
#         # logit_base_predicate[blank_mask] = torch.Tensor([0] * 3 + [1] + [0] * 48)
#         # logit_base_predicate[blank_mask].data = torch.Tensor([0] * 50 + [1]).unsqueeze(0).repeat(cut, 1)
#
#         for index in blank_mask:
#             logit_base_predicate[index, :] =  torch.Tensor([0] * 50 + [1]).unsqueeze(0)
#         pred_entry['rel_dists'] = logit_base_predicate
#     return pred_entry


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
    tmp = logit_base_predicate_51[mask_base_over_glat].view(-1, 51)
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

#
# def soft_merge(logit_base, logit_glat, node_type):
#
#     softmax_0 = torch.nn.Softmax(dim=0)
#     softmax_1 = torch.nn.Softmax(dim=1)
#
#     index = (node_type == 0).squeeze(0).unsqueeze(-1).repeat(1, 52)
#     logit_base_predicate = logit_base.data.squeeze(0)[index].view(-1, 52)
#
#     logit_base_predicate_51 = Variable(logit_base_predicate[:, 1:]).clone()
#     logit_glat_predicate_51 = logit_glat[:, 1:].clone()
#     logit_base_predicate[:, 1:] = softmax_1(logit_base_predicate_51).data
#     logit_glat[:, 1:] = softmax_1(logit_glat_predicate_51).data
#
#     logit_base_predicate_weight = torch.max(logit_base_predicate[:, 1:], dim=1)[0]
#     logit_glat_predicate_weight = torch.max(logit_glat[:, 1:], dim=1)[0]
#     combined_weight = torch.cat((logit_base_predicate_weight.unsqueeze(0),
#                                  logit_glat_predicate_weight.data.unsqueeze(0)), 0)
#     combined_weight = softmax_0(Variable(combined_weight))
#
#     logit_base_predicate_weight = combined_weight[0,:].unsqueeze(-1).repeat(1, 51)
#     logit_glat_predicate_weight = combined_weight[1,:].unsqueeze(-1).repeat(1, 51)
#
#     ones = torch.ones(logit_base_predicate_weight.size()[0], 1)
#
#     logit_base_predicate_weight = torch.cat((ones.cuda(), logit_base_predicate_weight.data), 1)
#     logit_glat_predicate_weight = torch.cat((ones.cuda(), logit_glat_predicate_weight.data), 1)
#
#     logit_base_predicate = logit_base_predicate * logit_base_predicate_weight
#     logit_glat = logit_glat.data * logit_glat_predicate_weight
#
#     output_logit_predicate = logit_base_predicate + logit_glat
#
#     output_logit_51 = Variable(logit_base_predicate[:, 1:]).clone()
#
#     output_logit_predicate[:, 1:] = softmax_1(output_logit_51).data
#
#     return output_logit_predicate


def glat_wrapper(total_data):
    # Batch size assumed to be 1
    input_class, adjs_con, adjs_lbl, node_type, node_logit, node_logit_dists = my_collate(total_data)
    # input_class, adjs_con, adjs_lbl, node_type = my_collate(total_data)
    if torch.is_tensor(input_class):
        input_class = Variable(input_class)
    # if torch.is_tensor(node_logit):
    #     node_logit = Variable(node_logit)
    if not torch.is_tensor(node_type):
        node_type = node_type.data
    if torch.is_tensor(adjs_con):
        adj_con = Variable(adjs_con)

    pred_label, pred_connect = model(input_class, adj_con, node_type)
    # pred_label, pred_connect = model(input_class, adj_con, node_type, node_logit_dists)
    # pred_label, pred_connect = model(input_class, adj_con, node_type, node_logit)

    # pred_label_predicate = input_class[node_type == 0]
    # pred_label_entities = input_class[node_type == 1]

    pred_label_predicate = pred_label[0]  # flatten predicate (B*N, 51)
    pred_label_entities = pred_label[1]  # flatten entities

    pred_label_predicate = soft_merge3(node_logit_dists, pred_label_predicate, node_type)

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

    pred_entry['rel_scores'] = tensor2variable(pred_entry['rel_scores'])
    pred_entry['pred_classes'] = tensor2variable(pred_entry['pred_classes'])

    pred_entry['rel_classes'] = torch.max(pred_entry['rel_scores'][:, 1:], dim=1)[1].unsqueeze(1) + 1
    pred_entry['rel_classes'] = variable2tensor(pred_entry['rel_classes'])
    pred_entry['pred_relations'] = torch.cat((pred_entry['pred_rel_inds'], pred_entry['rel_classes']), dim=1)

    # pred_entry = soft_merge6_original(pred_entry)

    # total_data = build_graph_structure(pred_entry, ind_to_classes, ind_to_predicates, if_predicting=if_predicting)

    # For SGCLS
    if conf.mode == "sgcls" or conf.mode == "sgdet":
        total_data, useless_entity_id = build_graph_structure(pred_entry, ind_to_classes, ind_to_predicates,
                                                              if_predicting=if_predicting,
                                                              sgclsdet=True)
    else:
        total_data = build_graph_structure(pred_entry, ind_to_classes, ind_to_predicates, if_predicting=if_predicting)

    pred_label_predicate, pred_label_entities = glat_wrapper(total_data)

    pred_entry['rel_scores'] = pred_label_predicate

    # For SGCLS
    pred_entry['entity_scores'] = pred_label_entities
    pred_entry['pred_classes'] = pred_label_entities.max(1)[1]

    # if "predcls" not in conf.mode:
    #     pred_entry['obj_scores'] = pred_label_entities

    # return pred_entry
    if conf.mode == "sgcls" or conf.mode == "sgdet":
        return pred_entry, useless_entity_id
    else:
        return pred_entry

all_pred_entries = []


def check_n_save(counter, det_res_list):
    length = len(det_res_list)
    threshold = 100
    if len(det_res_list) >= threshold:
        np.save("intermediate_" + str(threshold) + "/" + conf.model_s_m + '_val_output_' + str(counter) + '.npy', det_res_list)
        det_res_list = []
        counter += 1
        print("saving: " + conf.model_s_m + '_val_output_' + str(counter) + '.npy')
        print("counter: ", counter)
    print("len(det_res_list): ", length)
    return counter, det_res_list


# def val_batch(batch_num, b, evaluator,evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list, counter, det_res_list):
def val_batch(batch_num, b, evaluator, evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list):

    det_res = detector[b]

    if conf.num_gpus == 1:
        det_res = [det_res]

    for i, det in enumerate(det_res):

        if len(det) == 6 and not conf.return_top100:
            (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i, rel_dists) = det
            rels_i_a100 = np.asarray([])
        else:
            (boxes_i, objs_i, obj_scores_i, rels_i_b100, pred_scores_i_b100, rels_i_a100, pred_scores_i_a100,
             rel_scores_idx_b100, rel_scores_idx_a100, rel_dists) = det

    # det_res_list.append(list(det_res))
    # counter, det_res_list = check_n_save(counter, det_res_list)

    # return counter, det_res_list

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
                'rel_dists': rel_dists.data.cpu().numpy()
            }
        else:
            pred_entry = {
                'pred_boxes': boxes_i * BOX_SCALE / IM_SCALE,  # (23, 4) (16, 4)
                'pred_classes': objs_i,  # (23,) (16,)
                'pred_rel_inds': rels_i,  # (506, 2) (240, 2)
                'obj_scores': obj_scores_i,  # (23,) (16,)
                'rel_scores': pred_scores_i,  # hack for now. (506, 51) (240, 51)
                'rel_dists': rel_dists.data.cpu().numpy()
            }

        pred_entry = glat_postprocess(pred_entry, if_predicting=True)
        # pred_entry = cuda2numpy_dict(pred_entry)

        # For SGCLS
        if conf.mode == "sgcls" or conf.mode == "sgdet":
            useless_entity_id = pred_entry[1]
            pred_entry = pred_entry[0]

        # without adding a_100

        pred_entry['rel_scores'] = pred_entry['rel_scores'][:, :-1]

        # conditioning on adding a_100

        # if len(rels_i_a100.shape) == 1:
        #     pred_entry['rel_scores'] = pred_entry['rel_scores'][:, :-1]
        # else:
        #
        #     pred_entry['pred_rel_inds'] = np.concatenate((pred_entry['pred_rel_inds'], rels_i_a100), axis=0)
        #     pred_entry['rel_scores'] = np.concatenate((pred_entry['rel_scores'][:, :-1], pred_scores_i_a100), axis=0)

        all_pred_entries.append(pred_entry)

        # evaluator[conf.mode].evaluate_scene_graph_entry(
        #     gt_entry,
        #     pred_entry,
        # )

        eval_entry(conf.mode, gt_entry, pred_entry, evaluator, evaluator_multiple_preds,
                   evaluator_list, evaluator_multiple_preds_list)

# def val_batch(batch_num, b, evaluator, thrs=(20, 50, 100)):
#     det_res = detector[b]
#     if conf.num_gpus == 1:
#         det_res = [det_res]
#
#     for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):
#         gt_entry = {
#             'gt_classes': val.gt_classes[batch_num + i].copy(),
#             'gt_relations': val.relationships[batch_num + i].copy(),
#             'gt_boxes': val.gt_boxes[batch_num + i].copy(),
#         }
#         assert np.all(objs_i[rels_i[:,0]] > 0) and np.all(objs_i[rels_i[:,1]] > 0)
#         # assert np.all(rels_i[:,2] > 0)
#
#         pred_entry = {
#             'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
#             'pred_classes': objs_i,
#             'pred_rel_inds': rels_i,
#             'obj_scores': obj_scores_i,
#             'rel_scores': pred_scores_i,
#         }
#         all_pred_entries.append(pred_entry)
#
#         evaluator[conf.mode].evaluate_scene_graph_entry(
#             gt_entry,
#             pred_entry,
#         )

evaluator = BasicSceneGraphEvaluator.all_modes()
evaluator_multiple_preds = BasicSceneGraphEvaluator.all_modes(multiple_preds=True)
evaluator_list = []  # for calculating recall of each relationship except no relationship
evaluator_multiple_preds_list = []

evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=False)
for index, name in enumerate(ind_to_predicates):
    if index == 0:
        continue
    evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))
    evaluator_multiple_preds_list.append((index, name, BasicSceneGraphEvaluator.all_modes(multiple_preds=True)))

# if conf.cache is not None and os.path.exists(conf.cache):
#     print("Found {}! Loading from it".format(conf.cache))
#     with open(conf.cache,'rb') as f:
#         all_pred_entries = pkl.load(f)
#     for i, pred_entry in enumerate(tqdm(all_pred_entries)):
#         gt_entry = {
#             'gt_classes': val.gt_classes[i].copy(),
#             'gt_relations': val.relationships[i].copy(),
#             'gt_boxes': val.gt_boxes[i].copy(),
#         }
#         eval_entry(conf.mode, gt_entry, pred_entry, evaluator, evaluator_multiple_preds,
#                    evaluator_list, evaluator_multiple_preds_list)
#
#     recall = evaluator[conf.mode].print_stats()
#     recall_mp = evaluator_multiple_preds[conf.mode].print_stats()
#
#     mean_recall = calculate_mR_from_evaluator_list(evaluator_list, conf.mode, save_file=conf.save_rel_recall)
#     mean_recall_mp = calculate_mR_from_evaluator_list(evaluator_multiple_preds_list, conf.mode, multiple_preds=True,
#                                                       save_file=conf.save_rel_recall)
# else:
# det_res_list = []
# counter = 0
for val_b, batch in enumerate(tqdm(val_loader)):
# for val_b, batch in enumerate(val_loader):
    # total = len(val_loader)
    # print(val_b, " / ", total)
    # val_batch(conf.num_gpus*val_b, batch, evaluator)
    # counter, det_res_list = val_batch(conf.num_gpus * val_b, batch, evaluator, evaluator_multiple_preds, evaluator_list,
    #           evaluator_multiple_preds_list, counter, det_res_list)

    val_batch(conf.num_gpus * val_b, batch, evaluator, evaluator_multiple_preds, evaluator_list,
                                  evaluator_multiple_preds_list)
    torch.cuda.empty_cache()

# np.save(conf.model_s_m + '_val_output_1.npy', det_res_list)

recall = evaluator[conf.mode].print_stats()
recall_mp = evaluator_multiple_preds[conf.mode].print_stats()

mean_recall = calculate_mR_from_evaluator_list(evaluator_list, conf.mode, save_file=conf.save_rel_recall)
mean_recall_mp = calculate_mR_from_evaluator_list(evaluator_multiple_preds_list, conf.mode, multiple_preds=True,
                                                  save_file=conf.save_rel_recall)
# evaluator[conf.mode].print_stats()

if conf.cache is not None:
    with open(conf.cache,'wb') as f:
        pkl.dump(all_pred_entries, f)