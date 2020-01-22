#!/usr/bin/env bash

#Motif Predcls Mask with Finetuned glat

echo "EVALING MOTIFNET_MASK FINETUNED"
python models/eval_stanford_motif_glat_mask.py -m predcls -model_s_m motifnet -b 1 -clip 5 \
    -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/motifnet/vgrel-motifnet-sgcls.tar -nepoch 50 -cache ./cache/motifnet_glat_predcls_mask_nobg_pretrained | tee ./logs/eval_motif_glat_predcls_mask_nobg_pretrained.txt

#Motif Predcls Mask with pretrained glat
#echo "EVALING MOTIFNET_MASK PRETRAINED"
#python models/eval_stanford_motif_glat_mask.py -m predcls -model_s_m motifnet -b 1 -clip 5 \
#    -p 100 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/motifnet/vgrel-motifnet-sgcls.tar -nepoch 50 -cache ./cache/motifnet_glat_predcls_mask_pretrained | tee ./logs/eval_motif_glat_predcls_mask_pretrained_1.txt