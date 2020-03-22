#!/usr/bin/env bash

#Motif Predcls Mask with Finetuned glat

#echo "EVALING MOTIFNET_MASK FINETUNED V2"
python models/eval_rels_glat_visfea.py -m sgcls -model_s_m motifnet -b 1 -clip 5 \
    -p 100 -pooling_dim 4096 -ngpu 1 -test -ckpt checkpoints/motifnet/vgrel-motifnet-sgcls.tar -nepoch 50 -cache ./cache/motifnet_glat_visfea_sgcls | tee ./logs/eval_motif_glat_sgcls_visfea.txt

#Motif Predcls Mask with pretrained glat
#echo "EVALING MOTIFNET_MASK BASED ON THRESHOLD PRETRAINED"
#python models/eval_stanford_motif_glat_mask_logit_thres_v01.py -m predcls -model_s_m motifnet -b 1 -clip 5 \
#    -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/motifnet/vgrel-motifnet-sgcls.tar -nepoch 50 -cache ./cache/motifnet_glat_predcls_mask_logit_thres_pretrained | tee ./logs/eval_motif_glat_predcls_mask_logit_thres_pretrained.txt