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
codebase = '../../'
sys.path.append(codebase)
exp_name = 'motif'
import torch.nn as nn


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
                    inter_fea=True
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


class bg_classifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(bg_classifier, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(self.in_dim, 128, bias=True)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, self.out_dim, bias=True)
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        return x



# Freeze all the motif model
for n, param in detector.named_parameters():
    param.requires_grad = False
print(print_para(detector), flush=True)
ckpt = torch.load(conf.ckpt)
print("Loading EVERYTHING from motifnet", conf.ckpt)
optimistic_restore(detector, ckpt['state_dict'])
detector.cuda()
start_epoch = -1


model_bgc = bg_classifier(in_dim=4147, out_dim=2)
model_bgc.cuda()
optimizer_bgc = optim.Adam(model_bgc.parameters(), weight_decay=conf.adamwd, lr=conf.lr, eps=1e-3)
scheduler_bgc = lr_scheduler.StepLR(optimizer_bgc, step_size=5, gamma=0.3, last_epoch=start_epoch)


def train_epoch(epoch_num):
    detector.train()
    for n, param in detector.named_parameters():
        param.requires_grad = False
    model.train()
    model_bgc.train()
    tr = []
    start = time.time()
    accs = [0, 0]
    for b, batch in enumerate(train_loader):
        # res = train_batch(batch, accs=accs, verbose=b % (conf.print_interval*10) == 0)
        loss = train_batch(batch, accs=accs, verbose=b % (conf.print_interval*10) == 0)

        if b % conf.print_interval == 0 and b >= conf.print_interval:
            time_per_batch = (time.time() - start) / conf.print_interval
            print("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
                epoch_num, b, len(train_loader), time_per_batch, len(train_loader) * time_per_batch / 60))
            print('loss of mask:{:.3f}', loss)
            print('acc of mask:{:.3f}', accs[0]*1.0 / accs[1])
            print('num of right:', accs[0])
            print('-----------', flush=True)
            start = time.time()
            # break

    print("overall{:2d}: ({:.3f})\n".format(epoch, accs[0]*1.0 / accs[1]), flush=True)

    if use_tb:
        writer.add_scalar('loss/train_loss', loss, epoch)
        writer.add_scalar('acc/train_acc', accs[0]*1.0 / accs[1], epoch)

def train_batch(b, accs, verbose=False):
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
    result, inter_fea, det_res = detector[b]
    # result.rm_obj_dists(num_entities, 151)  result.obj_preds(num_entities)  result.rm_obj_labels(num_entities)
    # result.rel_dists(num_predicates, 51)  result.rel_labels(num_predicates)
    # result.rel_inds(num_predicates, 4)
    # pdb.set_trace()


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
    # pdb.set_trace()
    b_100_idx = det_res[-2]

    inter_fea = inter_fea[b_100_idx]
    gt_label = result.rel_labels[:, -1][b_100_idx]
    fg_idx = torch.nonzero(gt_label)
    if len(fg_idx) == 0:
        loss = -1
        return loss
    fg_idx = fg_idx[:, 0].data
    fg_number = len(fg_idx)
    bg_number = fg_number
    bg_idx = torch.nonzero(gt_label==0)[:bg_number, 0].data

    fg_rel_inds = det_res[3][fg_idx]
    fg_rel_scores = det_res[4][fg_idx]
    fg_rel_fea = inter_fea[fg_idx]

    bg_rel_inds = det_res[3][bg_idx]
    bg_rel_scores = det_res[4][bg_idx]
    bg_rel_fea = inter_fea[bg_idx]

    fg_rel_fea = torch.cat([fg_rel_fea, fg_rel_scores], dim=1)
    bg_rel_fea = torch.cat([bg_rel_fea, bg_rel_scores], dim=1)
    all_rel_fea = torch.cat([fg_rel_fea, bg_rel_fea], dim=0)

    all_rel_pred = model_bgc(all_rel_fea)

    gt_label = torch.zeros_like(result.rel_labels[:, -1])
    gt_label = gt_label[:(fg_number+bg_number)]
    gt_label[:fg_number] = 1

    loss = F.cross_entropy(all_rel_pred, gt_label)
    optimizer_bgc.zero_grad()
    loss.backward()
    optimizer_bgc.step()

    right_num = torch.sum(all_rel_pred.max(1)[1] == gt_label).data.cpu()[0]
    total_num = gt_label.size(0)
    accs[0] += right_num
    accs[1] += total_num
    # res = pd.Series({x: y.data[0] for x, y in loss.items()})
    return sum(loss)


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



def val_epoch(epoch):

    detector.eval()
    model.eval()
    model_bgc.eval()

    right_num_test = [0]*2
    total_num_test = [0]*2
    for val_b, batch in enumerate(val_loader):
        val_batch(conf.num_gpus * val_b, batch, right_num_test, total_num_test)

    print('-----------Finish testing epoch {}-----------'.format(epoch))
    print('test acc of mask bg:', right_num_test[0] * 1.0 / total_num_test[0])
    print('test acc of mask fg:', right_num_test[1] * 1.0 / total_num_test[1])
    if use_tb:
        writer.add_scalar('acc/test_acc_fg', right_num_test[1] * 1.0 / total_num_test[1], epoch)
        writer.add_scalar('acc/test_acc_bg', right_num_test[0] * 1.0 / total_num_test[0], epoch)


def val_batch(batch_num, b, right_num_test, total_num_test):
    dict_gt, inter_fea, det_res = detector[b]

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

        # pred_entry_init = copy.deepcopy(pred_entry)

        # pdb.set_trace()
        b_100_idx = torch.from_numpy(rel_scores_idx_b100).cuda()
        inter_fea = inter_fea[b_100_idx]
        all_rel_fea = torch.cat((inter_fea, Variable(torch.from_numpy(pred_scores_i_b100).cuda())), 1)
        all_rel_pred = model_bgc(all_rel_fea).data.cpu().numpy()
        gt_label = np.zeros(all_rel_pred.shape[0])

        # pdb.set_trace()

        for key, value in dict_gt.items():
            sub_idx = np.where(rels_i_b100[:, 0] == key[0])[0].tolist()
            obj_idx = np.where(rels_i_b100[:, 1] == key[1])[0].tolist()

            fg_idx = set(sub_idx) & set(obj_idx)
            if len(fg_idx) == 0:
                continue
            fg_idx = fg_idx.pop()
            gt_label[fg_idx] = 1

        num_class = 2
        for i in range(num_class):
            i_idx = np.where(gt_label==i)
            right_num_test[i] += np.sum(all_rel_pred[i_idx] == i)
            total_num_test[i] += len(i_idx)


print("Training starts now!")
# optimizer = get_optim(conf.lr * conf.num_gpus * conf.batch_size)

for epoch in range(start_epoch + 1, start_epoch + 1 + conf.num_epochs):
    print("start training epoch: ", epoch)
    scheduler_bgc.step()
    train_epoch(epoch)

    print("start validation epoch: ", epoch)
    val_epoch(epoch)
    # if use_tb:
    #     for key, value in recall.items():
    #         writer.add_scalar('eval_' + conf.mode + '_with_constraint/' + key, value, epoch)
    #     for key, value in recall_mp.items():
    #         writer.add_scalar('eval_' + conf.mode + '_without_constraint/' + key, value, epoch)
    #     for key, value in mean_recall.items():
    #         writer.add_scalar('eval_' + conf.mode + '_with_constraint/mean ' + key, value, epoch)
    #     for key, value in mean_recall_mp.items():
    #         writer.add_scalar('eval_' + conf.mode + '_without_constraint/mean ' + key, value, epoch)

