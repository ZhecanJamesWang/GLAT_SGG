#!/usr/bin/env bash
python models/train_rels_glat_visfea.py -m sgcls -p 400  -model_s_m kern -clip 5  \
-tb_log_dir summaries/kern_glat_visfea_train \
-save_dir checkpoints/kern_glat_visfea_train \
-ckpt checkpoints/kern_sgcls_predcls.tar \
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
| tee ./logs/train_kern_glat_sgcls_visfea.txt \



