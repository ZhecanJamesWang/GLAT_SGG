#!/usr/bin/env bash
python models/train_rels_glat_universal_v3.py -m sgdet -p 400  -model_s_m kern -clip 5  \
-tb_log_dir summaries/train_glat_kern_debiasedv1_sgdet_epoch1_v31 \
-save_dir checkpoints/train_glat_kern_debiasedv1_sgdet_epoch1_v31 \
-ckpt checkpoints/kern_balv1_sgdet/vgrel-17.tar \
-val_size 5000 \
-adam \
-b 12 \
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
| tee ./logs/train_glat_kern_debiasedv1_sgdet_epoch1_v31.txt \



