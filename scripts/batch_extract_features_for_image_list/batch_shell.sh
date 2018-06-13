#!/bin/bash

export PYTHONPATH=/usr/local/lib/python2.7/dist-packages:$PYTHON
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

MODEL_ROOT="/workspace/data/insightface-models/"
#change your model name
MODEL_NAME="model-r100-spa-m2.0/slim/model-r100-slim,96"
# MODEL_PATH="${MODEL_ROOT}${MODEL_NAME}"
echo $MODEL_NAME
#feature save
SAVE_ROOT="/workspace/data/shiyongjie/model/"
SAVE_NAME="model-r100-spa-m2.5-8gpu-ep32"
SAVE_DIR="${SAVE_ROOT}${SAVE_NAME}"

#log
logPath="./logs"
if [ ! -d "$logPath"];then
   mkdir  "$logPath"
fi

LOG_NAME="./logs/nohup-extract-log-ms1m-vggface2-model-r100-spa-m2.5-8gpu-ep32"
imagelistprefix="/workspace/data/shiyongjie/project/tron/megaface-challenge2-aligned-face-filelist-split"
GPU_NUM=8

for ((i=0; i<$GPU_NUM; i++))
    do
        /bin/sleep 2
        nohup python extract_features_for_aligned_imagelist.py \
              --model=$MODEL_NAME \
              --image-list=$imagelistprefix${i}.txt \
              --image-dir=/workspace/data/qyc \
              --save-dir=$SAVE_DIR \
              --batch-size=50 \
              --image-size=3,112,112 \
              --add-flip \
              --gpu=$i \
              --save-format=.npy \
              --flip-sim > ${LOG_NAME}"split"${i}".txt" 2>&1 &
    done