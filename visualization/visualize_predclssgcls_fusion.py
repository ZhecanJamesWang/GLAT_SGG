#-*- coding: utf-8 -*-
# visualization code for sgcls task
# in KERN root dir, run python visualization/visualize_sgcls.py -cache_dir caches/kern_sgcls.pkl -save_dir visualization/saves
from dataloaders.visual_genome import VGDataLoader, VG
# from dataloaders.VRD import VRDDataLoader, VRD
from graphviz import Digraph
import numpy as np
import torch
from tqdm import tqdm
from config import ModelConfig
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import dill as pkl
from collections import defaultdict
import gc 
import os
# conf = ModelConfig()
from config import VG_IMAGES, IM_DATA_FN, VG_SGG_FN, VG_SGG_DICT_FN, BOX_SCALE, IM_SCALE, PROPOSAL_FN
import argparse
import pdb

from torch.nn import functional as F
from torch.autograd import Variable



# base_model_path = "cache/motifnet_sgcls"
# fusion_model_path = "/home/tangtangwzc/KERN/cache/motif_predcls_2020_0224_0251"
# save_file_name = "visualization/motif_sgcls_baseline_fusion"


# base_model_path = "cache/stanford_sgcls"
# glat_model_path = "/home/tangtangwzc/KERN/cache/eval_stanford_glat_sgcls_2020_0227_0252_extra"
# fusion_model_path = "/home/tangtangwzc/KERN/cache/eval_stanford_glat_sgcls_2020_0227_0252"
# save_file_name = "visualization/stanford_sgcls_baseline_fusion_v3"

# base_model_path = "cache/stanford_sgcls"
# glat_model_path = "/home/tangtangwzc/KERN/cache/eval_stanford_glat_sgcls_2020_0311_1804_extra"
# fusion_model_path = "/home/tangtangwzc/KERN/cache/eval_stanford_glat_sgcls_2020_0311_1804"
# save_file_name = "visualization/stanford_sgcls_baseline_fusion_v2"


# caches/kern_sgcls_2020_0312_1153_extra

base_model_path = "/home/tangtangwzc/KERN/caches/kern_sgcls_2020_0303_1153.pkl"
glat_model_path = "/home/tangtangwzc/KERN/caches/kern_sgcls_2020_0312_1153.pkl_extra"
fusion_model_path = "/home/tangtangwzc/KERN/caches/kern_sgcls_2020_0312_1153.pkl_extra"
save_file_name = "visualization/kern_sgcls_baseline_fusion_v2"

# base_model_path = "cache/motifnet_sgcls"
# glat_model_path = "/home/tangtangwzc/KERN/cache/motif_predcls_2020_0228_1121_extra"
# fusion_model_path = "/home/tangtangwzc/KERN/cache/motif_predcls_2020_0228_1121"
# save_file_name = "visualization/motif_sgcls_baseline_fusion_v2"


parser = argparse.ArgumentParser(description='visualization for sgcls task')

parser.add_argument(
    '-save_dir',
    dest='save_dir',
    help='dir to save visualization files',
    type=str,
    default=save_file_name
)

# parser.add_argument(
#     '-save_dir',
#     dest='save_dir',
#     help='dir to save visualization files',
#     type=str,
#     default='visualization/motif_sgcls_baseline_mask_1'
# )

parser.add_argument(
    '-cache_dir_baseline',
    dest='cache_dir_baseline',
    help='dir to load cache sgcls results',
    type=str,
    default=base_model_path
)

parser.add_argument(
    '-cache_dir_change',
    dest='cache_dir_change',
    help='dir to load cache sgcls results',
    type=str,
    default=fusion_model_path
)

parser.add_argument(
    '-cache_dir_glat',
    dest='cache_dir_glat',
    help='dir to load cache sgcls results',
    type=str,
    default=glat_model_path
)


# parser.add_argument(
#     '-cache_dir_change',
#     dest='cache_dir_change',
#     help='dir to load cache sgcls results',
#     type=str,
#     default='cache/motifnet_glat_sgcls_mask_logit_thres_fintuned_v3_0.35_1',
#     # default='caches/kern_glat_sgcls_mask_logit_thres_v3_0.35_1.pkl',
# )


args = parser.parse_args()
os.makedirs(args.save_dir, exist_ok=True)
image_dir = os.path.join(args.save_dir, 'images')
graph_dir = os.path.join(args.save_dir, 'graphs')


os.makedirs(image_dir, exist_ok=True)
os.makedirs(graph_dir, exist_ok=True)
os.makedirs(os.path.join(graph_dir, 'ori'), exist_ok=True)
os.makedirs(os.path.join(graph_dir, 'glat'), exist_ok=True)


mode = 'sgcls' # this code is only for sgcls task

# train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
#                         use_proposals=conf.use_proposals,
#                         filter_non_overlap=conf.mode == 'sgdet')
train, val, test = VG.splits(num_val_im=5000, filter_duplicate_rels=True,
                        use_proposals=False,
                        filter_non_overlap=False)
val = test
# train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
#                                             batch_size=conf.batch_size,
#                                             num_workers=conf.num_workers,
#                                             num_gpus=conf.num_gpus)
train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                            batch_size=1,
                                            num_workers=1,
                                            num_gpus=1)
ind_to_predicates = train.ind_to_predicates
ind_to_classes = train.ind_to_classes

softmax_0 = torch.nn.Softmax(dim=0)
softmax_1 = torch.nn.Softmax(dim=1)


def soft_merge3(logit_base, logit_glat, obj=False, temp_model=1):

    logit_base_predicate = logit_base
    if obj:
        logit_glat = logit_glat[:, :-2]
    else:
        logit_glat = logit_glat[:, :-1]

    logit_base_predicate = softmax_1(Variable(logit_base_predicate)).data
    logit_glat_predicate = softmax_1(Variable(logit_glat)).data
    logit_base_predicate_one_hot = torch.max(logit_base_predicate[:, 1:-1], dim=1)[1]
    logit_base_predicate_weight = torch.max(logit_base_predicate, dim=1)[0]
    logit_glat_predicate_one_hot = torch.max(logit_glat_predicate[:, 1:-1], dim=1)[1]
    logit_glat_predicate_weight = torch.max(logit_glat_predicate, dim=1)[0] * temp_model
    combined_weight = torch.cat((logit_base_predicate_weight.unsqueeze(0), logit_glat_predicate_weight.unsqueeze(0)), 0)
    combined_weight = combined_weight/torch.sum(combined_weight, dim=0, keepdim=True)
    if obj == False:
        logit_base_predicate_weight = combined_weight[0,:].unsqueeze(-1).repeat(1, 51)
        logit_glat_predicate_weight = combined_weight[1,:].unsqueeze(-1).repeat(1, 51)
    else:
        logit_base_predicate_weight = combined_weight[0, :].unsqueeze(-1).repeat(1, 151)
        logit_glat_predicate_weight = combined_weight[1, :].unsqueeze(-1).repeat(1, 151)
    logit_base_predicate = logit_base_predicate * logit_base_predicate_weight
    logit_glat = logit_glat * logit_glat_predicate_weight

    return logit_base_predicate, logit_glat


def visualize_pred_gt(pred_entry_baseline, pred_entry_change, pred_entry_glat, gt_entry, ind_to_classes, ind_to_predicates, image_dir, graph_dir, top_k=50, save_format='pdf'):

    fn = gt_entry['fn']
    im = mpimg.imread(fn)
    max_len = max(im.shape)
    scale = BOX_SCALE / max_len
    fig, ax = plt.subplots(figsize=(12, 12))

    ax.imshow(im, aspect='equal')

    rois = gt_entry['gt_boxes']
    rois = rois / scale

    labels = gt_entry['gt_classes']
    # pred_classes = pred_entry['pred_classes']
    assert pred_entry_baseline['pred_classes'].shape == pred_entry_change['pred_classes'].shape
    pred_classes = pred_entry_glat['pred_label_entities_logit_base'][:, 1:].argmax(1) + 1
    pred_classes_glat = pred_entry_glat['pred_label_entities_logit_glat'][:, 1:-2].argmax(1) + 1
    pred_classes_change = pred_entry_glat['pred_label_entities_logit_fusion'][:, 1:].argmax(1) + 1

    # pred_classes = pred_entry_baseline['pred_classes']
    # pred_classes_change = pred_entry_change['pred_classes']
    # pred_classes_glat = pred_entry_glat['pred_classes']
    # --------------------------------------------------------------------

    # V2 compare logit during soft merge
    # obj_dists_change = pred_entry_glat["pred_label_entities_logit_fusion"]
    # obj_dists_ori = pred_entry_change["obj_scores_rm"]
    # obj_dists_glat = pred_entry_glat["pred_label_entities_logit_glat"]
    # obj_dists_fusion_ori, obj_dists_fusion_glat = soft_merge3(torch.from_numpy(obj_dists_ori), torch.from_numpy(obj_dists_glat), obj=True)
    # obj_scores_glat = F.softmax(Variable(obj_dists_fusion_glat[:, 1:])).data.numpy().max(1)
    # obj_scores_ori = F.softmax(Variable(obj_dists_fusion_ori[:, 1:])).data.numpy().max(1)

    # V1 compare output of baseline and glat model
    # pdb.set_trace()
    obj_dists_ori = pred_entry_glat['pred_label_entities_logit_base']
    obj_scores_ori = F.softmax(Variable(torch.from_numpy(obj_dists_ori[:, 1:]))).data.numpy().max(1)
    obj_dists_glat = pred_entry_glat['pred_label_entities_logit_glat'][:, :-2]
    obj_scores_glat = F.softmax(Variable(torch.from_numpy(obj_dists_glat[:, 1:]))).data.numpy().max(1)

    rels = gt_entry['gt_relations']

    # Filter out dupes!
    gt_rels = np.array(rels)
    # old_size = gt_rels.shape[0]
    all_rel_sets = defaultdict(list)
    for (o0, o1, r) in gt_rels:
        all_rel_sets[(o0, o1)].append(r)
    gt_rels = [(k[0], k[1], np.random.choice(v)) for k,v in all_rel_sets.items()]
    gt_rels = np.array(gt_rels)
    rels = gt_rels

    if rels.size > 0:
        rels_ = np.array(rels)
        rel_inds = rels_[:,:2].ravel().tolist()
    else:
        rel_inds = []
    object_name_list = []
    obj_count = np.zeros(151, dtype=np.int32)
    object_name_list_pred = []
    obj_count_pred = np.zeros(151, dtype=np.int32)
    for i, bbox in enumerate(rois):
        if int(labels[i]) == 0 or (i not in rel_inds):
            continue

        label_str = ind_to_classes[int(labels[i])]
        while label_str in object_name_list:
            obj_count[int(labels[i])] += 1
            label_str = label_str + '_' + str(obj_count[int(labels[i])])
        object_name_list.append(label_str)
        pred_classes_str = ind_to_classes[int(pred_classes_change[i])]
        while pred_classes_str in object_name_list_pred:
            obj_count_pred[int(pred_classes_change[i])] += 1
            pred_classes_str = pred_classes_str + '_' + str(obj_count_pred[int(pred_classes_change[i])])
        object_name_list_pred.append(pred_classes_str)
        if labels[i] == pred_classes_change[i]:
        # if labels[i] == pred_classes[i]:
            ax.text(bbox[0], bbox[1] - 2,
                    label_str,
                    bbox=dict(facecolor='green', alpha=0.5),
                    fontsize=32, color='white')
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1], fill=False,
                            edgecolor='green', linewidth=3.5)
            )
        else:
            ax.text(bbox[0], bbox[1] - 2,
                    pred_classes_str + ' ('+ label_str + ')',
                    bbox=dict(facecolor='red', alpha=0.5),
                    fontsize=32, color='white')
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1], fill=False,
                            edgecolor='red', linewidth=3.5)
                )

    # draw relations

    # pred_rel_inds = pred_entry['pred_rel_inds']
    # rel_scores = pred_entry['rel_scores']

    pred_rel_inds_ori = pred_entry_baseline['pred_rel_inds']
    pred_rel_inds_change = pred_entry_change['pred_rel_inds']
    pred_rel_inds_glat = pred_entry_glat['pred_rel_inds']

    # V2 compare logit during soft merge
    # rel_dists_ori = pred_entry_baseline['rel_scores']
    # rel_dists_glat = pred_entry_glat['pred_label_predicate_logit_glat']
    # rel_dists_fusion_ori, rel_dists_fusion_glat = soft_merge3(torch.from_numpy(rel_dists_ori), torch.from_numpy(rel_dists_glat), obj=False)
    # rel_dists_fusion_change = pred_entry_change['rel_scores']

    # V1 compare output of baseline and glat model
    # rel_dists_ori = pred_entry_baseline['rel_scores']
    # rel_dists_glat = pred_entry_glat['pred_label_predicate_logit_glat'][:,:-1]
    # rel_dists_change = pred_entry_change['rel_scores']

    # pdb.set_trace()

    rel_dists_ori = pred_entry_baseline['rel_dists']
    rel_dists_glat = pred_entry_glat['pred_label_predicate_logit_glat'][:,:-1]
    rel_dists_change = pred_entry_glat['pred_label_predicate_logit_fusion'][:,:-1]

    # pred_rels = np.column_stack((pred_rel_inds, 1+rel_scores[:,1:].argmax(1)))

    ax.axis('off')
    fig.tight_layout()

    # image_save_fn = os.path.join(image_dir, fn.split('/')[-1].split('.')[-2]+'.'+save_format)
    # plt.savefig(image_save_fn)
    # plt.close()


    sg_save_fn = os.path.join(graph_dir, fn.split('/')[-1].split('.')[-2])
    u = Digraph('sg', filename=sg_save_fn, format=save_format)
    u.attr('node', shape='box')
    u.body.append('size="6,6"')
    u.body.append('rankdir="LR"')

    name_list = []
    name_list_pred = []
    flag_has_node = False
    corrective_entity_number_ori = 0
    corrective_entity_number_glat = 0
    total_red_nodes = 0
    global total_change_entity_num

    for i, l in enumerate(labels):
        if i in rel_inds:
            name = ind_to_classes[l]
            name_pred = ind_to_classes[pred_classes[i]]
            name_suffix = 1
            # name_suffix_pred = 1
            obj_name = name
            obj_name_pred = name_pred
            obj_name_change = ind_to_classes[pred_classes_change[i]]
            obj_name_glat = ind_to_classes[pred_classes_glat[i]]
            obj_score_ori = obj_scores_ori[i]
            obj_score_glat = obj_scores_glat[i]
            # obj_score_change = obj_scores_change[i]
            while obj_name in name_list:
                obj_name = name + '_' + str(name_suffix)
                name_suffix += 1
            # while obj_name_pred in name_list_pred:
            #     obj_name_pred = name_pred + '_' + str(name_suffix_pred)
            #     name_suffix_pred += 1
            name_list.append(obj_name)
            name_list_pred.append(obj_name_pred)


            if pred_classes[i] == pred_classes_glat[i]:
                assert obj_name_pred == obj_name_glat
                assert pred_classes_glat[i] == pred_classes_change[i]
                if pred_classes_change[i] == l:
                    u.node(str(i), label=obj_name, color='black')
                else:
                    u.node(str(i), label=obj_name_pred+' ('+obj_name+')', color='red')
                    total_red_nodes += 1

            else:
                if pred_classes_change[i] == l:
                    # u.node(str(i), label='perception: ' + obj_name_pred + ' {:.2}'.format(obj_score_ori)+'\n'
                    #                                        + 'commonsense: ' + obj_name_glat + ' {:.2}'.format(obj_score_glat)
                    #                                        + '\n' + 'decision: ' + obj_name_change, color='forestgreen')
                    u.node(str(i), label='perception: ' + obj_name_pred +'\n'
                                                           + 'commonsense: ' + obj_name_glat
                                                           + '\n' + 'decision: ' + obj_name_change, color='forestgreen')
                    if pred_classes_glat[i] == l:
                        corrective_entity_number_glat += 1
                    else:
                        corrective_entity_number_ori += 1

                else:
                    u.node(str(i), label='perception: ' + obj_name_pred +'\n'
                                                           + 'commonsense: ' + obj_name_glat
                                                           + '\n' + 'decision: ' + obj_name_change + ' (' + obj_name + ')', color='red')
                    # u.node(str(i), label='perception: ' + obj_name_pred + ' {:.2}'.format(obj_score_ori)+'\n'
                    #                                        + 'commonsense: ' + obj_name_glat + ' {:.2}'.format(obj_score_glat)
                    #                                        + '\n' + 'decision: ' + obj_name_change + ' (' + obj_name + ')', color='red')
                    total_red_nodes += 1

            # if pred_classes[i] == pred_classes_change[i]:
            #     u.node(str(i), label=obj_name, color='black')
            # else:
            #     total_change_entity_num += 1
            #
            #     if pred_classes_change[i] == l:
            #         corrective_entity_number += 1
            #         u.node(str(i), label='perception:' + obj_name_pred + '\n'
            #                                                + 'commonsense:' + obj_name_change
            #                                                + '\n' + 'decision:' + obj_name_change, color='forestgreen')
            #     else:
            #         u.node(str(i), label='perception:' + obj_name_pred + '\n'
            #                                                + 'commonsense:' + obj_name_change
            #                                                + '\n' + 'decision:' + obj_name_change + '(' + obj_name + ')', color='red')


            # if l == pred_classes[i]:
            #     if pred_classes[i] == pred_classes_change[i]:
            #         u.node(str(i), label=obj_name, color='forestgreen')
            #     else:
            #         u.node(str(i), label=obj_name_pred + "->mask->" + obj_name_change+"("+obj_name+")", color='red')
            #         total_change_entity_num += 1
            #
            # else:
            #     if l == pred_classes_change[i]:
            #         u.node(str(i), label=obj_name_pred + "->mask->" + obj_name_change +"("+obj_name+")", color='blue')
            #         corrective_entity_number += 1
            #         total_change_entity_num+=1
            #
            #     else:
            #         u.node(str(i), label=obj_name_pred + "->mask->" + obj_name_change +"("+obj_name+")", color='red')
            #         if pred_classes_change[i] != pred_classes[i]:
            #             total_change_entity_num += 1

            flag_has_node = True
    if not flag_has_node:
        return

    # pred_rels = np.column_stack((pred_rel_inds, 1+rel_scores[:,1:].argmax(1)))
    # pred_rels = pred_rels[:top_k]

    pred_rels_ori = np.column_stack((pred_rel_inds_ori, 1+rel_dists_ori[:,1:].argmax(1)))
    pred_rels_change = np.column_stack((pred_rel_inds_change, 1+rel_dists_change[:,1:].argmax(1)))
    pred_rels_glat = np.column_stack((pred_rel_inds_glat, 1+rel_dists_glat[:,1:].argmax(1)))


    pred_rels_ori = pred_rels_ori[:top_k]
    pred_rels_change = pred_rels_change[:top_k]
    pred_rels_glat = pred_rels_glat[:top_k]

    # for pred_rel in pred_rels:
    #     for rel in rels:
    #         if pred_rel[0] == rel[0] and pred_rel[1] == rel[1]:
    #             if pred_rel[2] == rel[2]:
    #                 u.edge(str(rel[0]), str(rel[1]), label=ind_to_predicates[rel[2]], color='forestgreen')
    #             else:
    #                 label_str = ind_to_predicates[pred_rel[2]] + ' (' + ind_to_predicates[rel[2]] + ')'
    #                 u.edge(str(rel[0]), str(rel[1]), label=label_str, color='red')

    corrective_pred_number_ori=0
    corrective_pred_number_glat=0

    for idx, pred_rel_ori in enumerate(pred_rels_ori):
        rel_score_ori = (Variable(torch.from_numpy(rel_dists_ori[idx, 1:]))).data.numpy().max()
        rel_score_glat = F.softmax(Variable(torch.from_numpy(rel_dists_glat[idx, 1:]))).data.numpy().max()
        # pdb.set_trace()
        for rel in rels:
            if pred_rel_ori[0] == rel[0] and pred_rel_ori[1] == rel[1]:

                if pred_rels_glat[idx, 2] == pred_rel_ori[2]:
                    assert pred_rels_glat[idx, 2] == pred_rels_change[idx, 2]
                    if pred_rels_change[idx, 2] == rel[2]:
                        u.edge(str(rel[0]), str(rel[1]), label=ind_to_predicates[pred_rel_ori[2]], color='black')
                    else:
                        u.edge(str(rel[0]), str(rel[1]), label=ind_to_predicates[pred_rel_ori[2]]+' ('+ind_to_predicates[rel[2]]+')', color='red')
                        total_red_nodes += 1

                else:
                    if pred_rels_change[idx, 2] == rel[2]:
                        # u.edge(str(rel[0]), str(rel[1]), label='perception: ' + ind_to_predicates[pred_rel_ori[2]] + ' {:.2}\n'.format(rel_score_ori)
                        #                                        + 'commonsense: ' + ind_to_predicates[pred_rels_glat[idx, 2]]
                        #                                        + ' {:.2}\n'.format(rel_score_glat) + 'decision: '+ ind_to_predicates[pred_rels_change[idx, 2]], color='forestgreen')
                        u.edge(str(rel[0]), str(rel[1]), label='perception: ' + ind_to_predicates[pred_rel_ori[2]] +'\n'
                                                               + 'commonsense: ' + ind_to_predicates[pred_rels_glat[idx, 2]] +'\n'
                                                               + 'decision: '+ ind_to_predicates[pred_rels_change[idx, 2]], color='forestgreen')
                        if pred_rels_ori[idx, 2] == rel[2]:
                            corrective_pred_number_ori += 1
                        elif pred_rels_glat[idx, 2] == rel[2]:
                            corrective_pred_number_glat += 1

                    else:
                        # u.edge(str(rel[0]), str(rel[1]), label='perception: ' + ind_to_predicates[pred_rel_ori[2]] + ' {:.2}\n'.format(rel_score_ori)
                        #                                        + 'commonsense: ' + ind_to_predicates[pred_rels_glat[idx, 2]] + ' {:.2}\n'.format(rel_score_glat)
                        #                                        + 'decision: '+ ind_to_predicates[pred_rels_change[idx, 2]] + ' (' + ind_to_predicates[rel[2]] + ')', color='red')
                        u.edge(str(rel[0]), str(rel[1]),
                               label='perception: ' + ind_to_predicates[pred_rel_ori[2]] + '\n'
                                     + 'commonsense: ' + ind_to_predicates[pred_rels_glat[idx, 2]] + '\n'
                                     + 'decision: ' + ind_to_predicates[pred_rels_change[idx, 2]] + ' (' +
                                     ind_to_predicates[rel[2]] + ')', color='red')

                        total_red_nodes += 1



                # if pred_rels_change[idx, 2] == pred_rel_ori[2]:
                #     u.edge(str(rel[0]), str(rel[1]), label=ind_to_predicates[pred_rel_ori[2]], color='black')
                # else:
                #     if pred_rels_change[idx, 2] == rel[2]:
                #         u.edge(str(rel[0]), str(rel[1]), label='perception:' + ind_to_predicates[pred_rel_ori[2]] + '\n'
                #                                                + 'commonsense:' + ind_to_predicates[pred_rels_change[idx, 2]]
                #                                                + '\n' + 'decision:'+ ind_to_predicates[pred_rels_change[idx, 2]], color='forestgreen')
                #         corrective_pred_number += 1
                #     else:
                #         u.edge(str(rel[0]), str(rel[1]), label='perception:' + ind_to_predicates[pred_rel_ori[2]] + '\n'
                #                                                + 'commonsense:' + ind_to_predicates[pred_rels_change[idx, 2]]
                #                                                + '\n' + 'decision:'+ ind_to_predicates[pred_rels_change[idx, 2]] + '(' + ind_to_predicates[rel[2]] + ')', color='red')


                # if pred_rel_ori[2] == rel[2]:
                #     if pred_rels_change[idx, 2] == rel[2]:
                #         u.edge(str(rel[0]), str(rel[1]), label=ind_to_predicates[rel[2]], color='forestgreen')
                #     else:
                #         u.edge(str(rel[0]), str(rel[1]), label=ind_to_predicates[pred_rel_ori[2]]+'->mask->'+ind_to_predicates[pred_rels_change[idx, 2]]+'('+ind_to_predicates[rel[2]]+')', color='red')
                #
                # else:
                #     if pred_rels_change[idx, 2] == rel[2]:
                #         label_str = ind_to_predicates[pred_rel_ori[2]] + '->mask->' + ind_to_predicates[pred_rels_change[idx, 2]]
                #         u.edge(str(rel[0]), str(rel[1]), label=label_str, color='blue')
                #         corrective_pred_number += 1
                #     else:
                #         label_str = ind_to_predicates[pred_rel_ori[2]] + '->mask->' + ind_to_predicates[pred_rels_change[idx, 2]] + '('+ ind_to_predicates[rel[2]] +')'
                #         u.edge(str(rel[0]), str(rel[1]), label=label_str, color='red')


    for rel in rels:
        flag_rel_has_pred = False
        for pred_rel_ori in pred_rels_ori:
            if pred_rel_ori[0] == rel[0] and pred_rel_ori[1] == rel[1]:
                flag_rel_has_pred = True
                break
        if not flag_rel_has_pred:
            u.edge(str(rel[0]), str(rel[1]), label=ind_to_predicates[rel[2]], color='grey50')

    graph_dir_ori = os.path.join(graph_dir, 'ori')
    graph_dir_glat = os.path.join(graph_dir, 'glat')


    total_cor_ori = corrective_entity_number_ori+corrective_pred_number_ori
    total_cor_glat = corrective_entity_number_glat+corrective_pred_number_glat
    total_change_entity_num += (corrective_entity_number_ori+corrective_entity_number_glat)

    if total_cor_ori > 0:
        u.render(filename=os.path.join(graph_dir_ori, str(total_red_nodes)+'_'+fn.split('/')[-1].split('.')[-2]), view=False, cleanup=True)

    if total_cor_glat > 0:
        u.render(filename=os.path.join(graph_dir_glat, str(total_red_nodes)+'_'+fn.split('/')[-1].split('.')[-2]), view=False, cleanup=True)

    if total_cor_ori > 0 or total_cor_glat> 0:
        image_save_fn = os.path.join(image_dir, fn.split('/')[-1].split('.')[-2]+'.'+save_format)
        plt.savefig(image_save_fn)
    plt.close()

with open(args.cache_dir_baseline, 'rb') as f:
    pred_entries_baseline = pkl.load(f)

with open(args.cache_dir_change, 'rb') as f:
    pred_entries_change = pkl.load(f)

with open(args.cache_dir_glat, 'rb') as f:
    pred_entries_glat = pkl.load(f)

total_change_entity_num = 0

for i, (pred_entry_baseline, pred_entry_change, pred_entry_glat) in enumerate(tqdm(zip(pred_entries_baseline, pred_entries_change, pred_entries_glat))):
    # if i == 2000:
    #     break
    gt_entry = {
        'gt_classes': val.gt_classes[i].copy(),
        'gt_relations': val.relationships[i].copy(),
        'gt_boxes': val.gt_boxes[i].copy(),
        'fn': val.filenames[i]
    }
    # you could use these three lines of code to only visualize some images
    # num_id = gt_entry['fn'].split('/')[-1].split('.')[-2]
    # if num_id == '2343586' or num_id == '2343599' or num_id == '2315539':
    #     visualize_pred_gt(pred_entry, gt_entry, ind_to_classes, ind_to_predicates, image_dir=image_dir, graph_dir=graph_dir, top_k=50)

    visualize_pred_gt(pred_entry_baseline, pred_entry_change, pred_entry_glat, gt_entry, ind_to_classes, ind_to_predicates, image_dir=image_dir, graph_dir=graph_dir, top_k=50)
    # visualize_pred_gt_ori(pred_entry_ori, gt_entry, ind_to_classes, ind_to_predicates, image_dir=image_dir, graph_dir=graph_dir, top_k=50)

print("total change entity number:", total_change_entity_num)

#
# for i, pred_entry_ori in enumerate(tqdm(pred_entries_predcls)):
#     # if i == 500:
#     #     break
#     gt_entry = {
#         'gt_classes': val.gt_classes[i].copy(),
#         'gt_relations': val.relationships[i].copy(),
#         'gt_boxes': val.gt_boxes[i].copy(),
#         'fn': val.filenames[i]
#     }
#     # you could use these three lines of code to only visualize some images
#     # num_id = gt_entry['fn'].split('/')[-1].split('.')[-2]
#     # if num_id == '2343586' or num_id == '2343599' or num_id == '2315539':
#     #     visualize_pred_gt(pred_entry, gt_entry, ind_to_classes, ind_to_predicates, image_dir=image_dir, graph_dir=graph_dir, top_k=50)
#     visualize_pred_gt_ori(pred_entry_ori, gt_entry, ind_to_classes, ind_to_predicates, image_dir=image_dir, graph_dir=graph_dir, top_k=50)
