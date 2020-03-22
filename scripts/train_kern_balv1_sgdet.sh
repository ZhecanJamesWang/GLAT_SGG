#!/usr/bin/env bash
python models/kern_bal.py -m sgdet -p 400 -clip 5 \
-tb_log_dir summaries/kern_balv1_sgdet \
-save_dir checkpoints/kern_balv1_sgdet \
-ckpt checkpoints/kern_bal_sgcls_v1/vgrel-16.tar \
-val_size 5000 \
-adam \
-b 2 \
-lr 1e-5 \
-reweight_matrix_ent prior_matrices/ent_counts.pkl \
-reweight_matrix_pred prior_matrices/pred_counts.pkl \
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
| tee ./logs/train_kern_balv1_sgdet.txt