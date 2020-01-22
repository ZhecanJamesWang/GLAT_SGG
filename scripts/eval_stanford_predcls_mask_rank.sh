#!/usr/bin/env bash

#Motif Predcls Mask with Finetuned glat

#echo "EVALING STANFORD_MASK FINETUNED"
#python models/eval_stanford_motif_glat_mask.py -m predcls -model_s_m stanford -b 1 -clip 5 \
#    -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/stanford_1/vgrel-7.tar -nepoch 50 -cache ./cache/motifnet_glat_predcls_mask | tee ./logs/eval_motif_glat_predcls_mask.txt

#Motif Predcls Mask with pretrained glat
echo "EVALING STANFORD_MASK PRETRAINED"
python models/eval_stanford_motif_glat_mask_rank.py -m predcls -model_s_m stanford -b 1 -clip 5 \
    -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/stanford_1/vgrel-7.tar -nepoch 50 -cache ./cache/stanford_glat_predcls_mask_rank_pretrained | tee ./logs/eval_stanford_glat_predcls_mask_rank_pretrained.txt