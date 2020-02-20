#!/usr/bin/env bash

#Motif Predcls Mask with Finetuned glat

#0.2, 0.25, 0.3, 0.35, 0.4,
#echo "EVALING MOTIFNET_MASK PRETRAINED"
python models/eval_stanford_motif_glat_mask_logit_thres_v2.py -threshold 0.2 -m predcls -model_s_m motifnet -b 1 -clip 5 \
    -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/motifnet/vgrel-motifnet-sgcls.tar -nepoch 50 -cache ./cache/motifnet_glat_predcls_mask_logit_thres_in100_0.2 | tee ./logs/eval_motif_glat_predcls_mask_logit_thres_in100_0.2.txt

python models/eval_stanford_motif_glat_mask_logit_thres_v2.py -threshold 0.25 -m predcls -model_s_m motifnet -b 1 -clip 5 \
    -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/motifnet/vgrel-motifnet-sgcls.tar -nepoch 50 -cache ./cache/motifnet_glat_predcls_mask_logit_thres_in100_0.25 | tee ./logs/eval_motif_glat_predcls_mask_logit_thres_in100_0.25.txt

python models/eval_stanford_motif_glat_mask_logit_thres_v2.py -threshold 0.3 -m predcls -model_s_m motifnet -b 1 -clip 5 \
    -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/motifnet/vgrel-motifnet-sgcls.tar -nepoch 50 -cache ./cache/motifnet_glat_predcls_mask_logit_thres_in100_0.3 | tee ./logs/eval_motif_glat_predcls_mask_logit_thres_in100_0.3.txt

python models/eval_stanford_motif_glat_mask_logit_thres_v2.py -threshold 0.35 -m predcls -model_s_m motifnet -b 1 -clip 5 \
    -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/motifnet/vgrel-motifnet-sgcls.tar -nepoch 50 -cache ./cache/motifnet_glat_predcls_mask_logit_thres_in100_0.35 | tee ./logs/eval_motif_glat_predcls_mask_logit_thres_in100_0.35.txt

python models/eval_stanford_motif_glat_mask_logit_thres_v2.py -threshold 0.4 -m predcls -model_s_m motifnet -b 1 -clip 5 \
    -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/motifnet/vgrel-motifnet-sgcls.tar -nepoch 50 -cache ./cache/motifnet_glat_predcls_mask_logit_thres_in100_0.4 | tee ./logs/eval_motif_glat_predcls_mask_logit_thres_in100_0.4.txt


#Motif Predcls Mask with pretrained glat
#echo "EVALING MOTIFNET_MASK BASED ON THRESHOLD PRETRAINED"
#python models/eval_stanford_motif_glat_mask_logit_thres_v01.py -m predcls -model_s_m motifnet -b 1 -clip 5 \
#    -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/motifnet/vgrel-motifnet-sgcls.tar -nepoch 50 -cache ./cache/motifnet_glat_predcls_mask_logit_thres_pretrained | tee ./logs/eval_motif_glat_predcls_mask_logit_thres_pretrained.txt