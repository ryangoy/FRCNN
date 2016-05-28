#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET MODEL DATASET ITERS [options args to {train,test}_net.py]
# DATASET is either pascal_voc, coco, imagenet, or ir.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG16 ./data/imagenet_models VGG16.v3.caffemodel imagenet 10000

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
MODEL=$3
DATASET=$4
ITERS=$5

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ;;
  coco)
    #TRAIN_IMDB="desktops_train"
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_minival"
    PT_DIR="coco"
    ;;
  imagenet)
    TRAIN_IMDB="desktops_train"
    TEST_IMDB="desktops_test"
    PT_DIR="imagenet"
    ;;
  ir)
    TRAIN_IMDB="ir_train"
    TEST_IMDB="ir_test"
    PT_DIR="imagenet"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/faster_rcnn_end2end_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
echo "$PT_DIR"
time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver ./models/${PT_DIR}/${NET}/faster_rcnn_end2end/solver.prototxt \
  --weights ${MODEL} \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg ./experiments/cfgs/faster_rcnn_end2end.yml \
  ${EXTRA_ARGS}
set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x
