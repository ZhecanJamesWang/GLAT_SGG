#!/usr/bin/env bash

#Motif Predcls Mask with Finetuned glat

echo "EVALING MOTIFNET_MASK RANK FINETUNED"
python models/eval_stanford_motif_glat_mask_rank.py -m predcls -model_s_m motifnet -b 1 -clip 5 \
    -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/motifnet/vgrel-motifnet-sgcls.tar -nepoch 50 -cache ./cache/motifnet_glat_predcls_mask_rank_fintuned_changemaskonly | tee ./logs/eval_motif_glat_predcls_mask_rank_fintuned_changemaskonly.txt

#Motif Predcls Mask with pretrained glat
#echo "EVALING MOTIFNET_MASK RANK PRETRAINED"
#python models/eval_stanford_motif_glat_mask_rank.py -m predcls -model_s_m motifnet -b 1 -clip 5 \
#    -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/motifnet/vgrel-motifnet-sgcls.tar -nepoch 50 -cache ./cache/motifnet_glat_predcls_mask_rank_pretrained | tee ./logs/eval_motif_glat_predcls_mask_rank_pretrained.txt