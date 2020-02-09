#!/usr/bin/env bash

#0.2, 0.25, 0.3, 0.35, 0.4, 0.05, 0.10, 0.15,


python models/eval_kern_mask_logit_thres.py -m predcls -p 100 -clip 5  -threshold 0.2 \
-ckpt checkpoints/kern_sgcls_predcls.tar \
-test \
-b 1 \
-use_ggnn_rel \
-ggnn_rel_time_step_num 3 \
-ggnn_rel_hidden_dim 512 \
-ggnn_rel_output_dim 512 \
-use_rel_knowledge \
-rel_knowledge prior_matrices/rel_matrix.npy \
-cache caches/kern_glat_predcls_mask_logit_thres_in30_0.2.pkl \
-save_rel_recall results/kern_glat_predcls_mask_logit_thres_in30_0.2_recall_predcls.pkl \
| tee ./logs/eval_kern_glat_predcls_mask_logit_thres_in30_0.2.txt \


python models/eval_kern_mask_logit_thres.py -m predcls -p 100 -clip 5  -threshold 0.25 \
-ckpt checkpoints/kern_sgcls_predcls.tar \
-test \
-b 1 \
-use_ggnn_rel \
-ggnn_rel_time_step_num 3 \
-ggnn_rel_hidden_dim 512 \
-ggnn_rel_output_dim 512 \
-use_rel_knowledge \
-rel_knowledge prior_matrices/rel_matrix.npy \
-cache caches/kern_glat_predcls_mask_logit_thres_in30_0.25.pkl \
-save_rel_recall results/kern_glat_predcls_mask_logit_thres_in30_0.25_recall_predcls.pkl \
| tee ./logs/eval_kern_glat_predcls_mask_logit_thres_in30_0.25.txt \


python models/eval_kern_mask_logit_thres.py -m predcls -p 100 -clip 5  -threshold 0.3 \
-ckpt checkpoints/kern_sgcls_predcls.tar \
-test \
-b 1 \
-use_ggnn_rel \
-ggnn_rel_time_step_num 3 \
-ggnn_rel_hidden_dim 512 \
-ggnn_rel_output_dim 512 \
-use_rel_knowledge \
-rel_knowledge prior_matrices/rel_matrix.npy \
-cache caches/kern_glat_predcls_mask_logit_thres_in30_0.3.pkl \
-save_rel_recall results/kern_glat_predcls_mask_logit_thres_in30_0.3_recall_predcls.pkl \
| tee ./logs/eval_kern_glat_predcls_mask_logit_thres_in30_0.3.txt \


python models/eval_kern_mask_logit_thres.py -m predcls -p 100 -clip 5  -threshold 0.35 \
-ckpt checkpoints/kern_sgcls_predcls.tar \
-test \
-b 1 \
-use_ggnn_rel \
-ggnn_rel_time_step_num 3 \
-ggnn_rel_hidden_dim 512 \
-ggnn_rel_output_dim 512 \
-use_rel_knowledge \
-rel_knowledge prior_matrices/rel_matrix.npy \
-cache caches/kern_glat_predcls_mask_logit_thres_in30_0.35.pkl \
-save_rel_recall results/kern_glat_predcls_mask_logit_thres_in30_0.35_recall_predcls.pkl \
| tee ./logs/eval_kern_glat_predcls_mask_logit_thres_in30_0.35.txt \


python models/eval_kern_mask_logit_thres.py -m predcls -p 100 -clip 5  -threshold 0.4 \
-ckpt checkpoints/kern_sgcls_predcls.tar \
-test \
-b 1 \
-use_ggnn_rel \
-ggnn_rel_time_step_num 3 \
-ggnn_rel_hidden_dim 512 \
-ggnn_rel_output_dim 512 \
-use_rel_knowledge \
-rel_knowledge prior_matrices/rel_matrix.npy \
-cache caches/kern_glat_predcls_mask_logit_thres_in30_0.4.pkl \
-save_rel_recall results/kern_glat_predcls_mask_logit_thres_in30_0.4_recall_predcls.pkl \
| tee ./logs/eval_kern_glat_predcls_mask_logit_thres_in30_0.4.txt \


python models/eval_kern_mask_logit_thres.py -m predcls -p 100 -clip 5  -threshold 0.05 \
-ckpt checkpoints/kern_sgcls_predcls.tar \
-test \
-b 1 \
-use_ggnn_rel \
-ggnn_rel_time_step_num 3 \
-ggnn_rel_hidden_dim 512 \
-ggnn_rel_output_dim 512 \
-use_rel_knowledge \
-rel_knowledge prior_matrices/rel_matrix.npy \
-cache caches/kern_glat_predcls_mask_logit_thres_in30_0.05.pkl \
-save_rel_recall results/kern_glat_predcls_mask_logit_thres_in30_0.05_recall_predcls.pkl \
| tee ./logs/eval_kern_glat_predcls_mask_logit_thres_in30_0.05.txt \


python models/eval_kern_mask_logit_thres.py -m predcls -p 100 -clip 5  -threshold 0.1 \
-ckpt checkpoints/kern_sgcls_predcls.tar \
-test \
-b 1 \
-use_ggnn_rel \
-ggnn_rel_time_step_num 3 \
-ggnn_rel_hidden_dim 512 \
-ggnn_rel_output_dim 512 \
-use_rel_knowledge \
-rel_knowledge prior_matrices/rel_matrix.npy \
-cache caches/kern_glat_predcls_mask_logit_thres_in30_0.1.pkl \
-save_rel_recall results/kern_glat_predcls_mask_logit_thres_in30_0.1_recall_predcls.pkl \
| tee ./logs/eval_kern_glat_predcls_mask_logit_thres_in30_0.1.txt \


python models/eval_kern_mask_logit_thres.py -m predcls -p 100 -clip 5  -threshold 0.15 \
-ckpt checkpoints/kern_sgcls_predcls.tar \
-test \
-b 1 \
-use_ggnn_rel \
-ggnn_rel_time_step_num 3 \
-ggnn_rel_hidden_dim 512 \
-ggnn_rel_output_dim 512 \
-use_rel_knowledge \
-rel_knowledge prior_matrices/rel_matrix.npy \
-cache caches/kern_glat_predcls_mask_logit_thres_in30_0.15.pkl \
-save_rel_recall results/kern_glat_predcls_mask_logit_thres_in30_0.15_recall_predcls.pkl \
| tee ./logs/eval_kern_glat_predcls_mask_logit_thres_in30_0.15.txt \
