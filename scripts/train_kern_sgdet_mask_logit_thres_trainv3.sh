#!/usr/bin/env bash
python models/train_rels_kern_mask_logit_thres_train_v3mbz_sgdet.py -m sgdet -p 500 -clip 5 -threshold 0.35 \
-tb_log_dir summaries/kern_glat_sgdet_mask_logit_thres_train_v3mbz_1 \
-save_dir checkpoints/kern_glat_sgdet_mask_logit_thres_train_v3mbz_1 \
-ckpt checkpoints/kern_sgdet.tar \
-val_size 5000 \
-adam \
-b 16 \
-lr 1e-4 \
-use_ggnn_obj \
-ggnn_obj_time_step_num 3 \
-ggnn_obj_hidden_dim 512 \
-ggnn_obj_output_dim 512 \
-use_obj_knowledge \
-obj_knowledge prior_matrices/obj_matrix.npy \
-use_ggnn_rel \
-ggnn_rel_time_step_num 3 \
-ggnn_rel_hidden_dim 512 \
-ggnn_rel_output_dim 512 \
-use_rel_knowledge \
-rel_knowledge prior_matrices/rel_matrix.npy \
| tee ./logs/train_kern_glat_sgdet_mask_logit_thres_train_v3mbz_1.txt \



