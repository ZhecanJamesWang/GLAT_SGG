#!/usr/bin/env bash

#Motif Predcls Mask with Finetuned glat

#echo "EVALING MOTIFNET_MASK FINETUNED"
#python models/eval_stanford_motif_glat_mask_threshold.py -m predcls -model_s_m motifnet -b 1 -clip 5 \
#    -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/motifnet/vgrel-motifnet-sgcls.tar -nepoch 50 -cache ./cache/motifnet_glat_predcls_mask_1 | tee ./logs/eval_motif_glat_predcls_mask_1.txt

#Motif Predcls Mask with pretrained glat
echo "EVALING MOTIFNET_MASK BASED ON THRESHOLD PRETRAINED"
python models/eval_stanford_motif_glat_mask_threshold.py -m predcls -model_s_m motifnet -b 1 -clip 5 \
    -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/motifnet/vgrel-motifnet-sgcls.tar -nepoch 50 -cache ./cache/motifnet_glat_predcls_mask_threshold_pretrained | tee ./logs/eval_motif_glat_predcls_mask_threshold_pretrained.txt