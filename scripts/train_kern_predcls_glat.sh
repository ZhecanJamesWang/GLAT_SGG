#!/usr/bin/env bash
python models/train_rels.py -m predcls -p 100 -clip 5 \
-tb_log_dir summaries/kern_predcls_glat \
-save_dir checkpoints/kern_predcls_glat \
-ckpt checkpoints/kern_sgcls_predcls.tar \
-val_size 5000 \
-adam \
-b 4 \
-lr 1e-4 \
-use_ggnn_rel \
-ggnn_rel_time_step_num 3 \
-ggnn_rel_hidden_dim 512 \
-ggnn_rel_output_dim 512 \
-use_rel_knowledge \
-rel_knowledge prior_matrices/rel_matrix.npy \
