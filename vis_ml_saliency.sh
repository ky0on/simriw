#!/bin/sh

MODEL=$1
python vis_ml_saliency.py -n=500 $MODEL
python vis_ml_saliency.py -n=500 --ATHHT=0.36 $MODEL
python vis_ml_saliency.py -n=500 --ATHLT=0.35 $MODEL
