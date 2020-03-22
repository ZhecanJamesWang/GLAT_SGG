#!/usr/bin/env bash


#echo "EVALING KERN With vis feature"
python models/eval_rels_glat_visfea.py -m sgcls -model_s_m kern -b 1 -clip 5 \
-p 100 \
-ngpu 1 \
-test \
-ckpt checkpoints/kern_sgcls_predcls.tar \
-nepoch 50 \
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
-save_rel_recall results/kern_glat_visfea_sgcls.pkl \
-cache ./cache/kern_glat_visfea_sgcls \
| tee ./logs/eval_kern_glat_sgcls_visfea.txt
