#!/bin/bash

export PYTHONPATH=/usr/local/lib/python2.7/dist-packages:$PYTHON
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

GPU_NUM=8
MODEL_ROOT="/workspace/data/insightface-models/"
#change your model name
MODEL_NAME="model-r100-spa-m2.0/slim/model-r100-slim,96"
MODEL_PATH="${MODEL_ROOT}${MODEL_NAME}"
echo $MODEL_PATH
#feature save
SAVE_ROOT="/workspace/data/shiyongjie/data/"
SAVE_NAME="megaface_challenge2_features"
SAVE_DIR="${SAVE_ROOT}${SAVE_NAME}"

#split feature file list
FEATURE_FILE_LIST="/workspace/data/shiyongjie/code/megaface2_filelist.txt"
SPLIT_FILE_LIST_SAVE_PATH="./filelist"
SPLIT_FILE_LIST_PREFIX="split"
DIVISION_NUM=$GPU_NUM

if [ ! -d $SPLIT_FILE_LIST_SAVE_PATH ];then
   mkdir $SPLIT_FILE_LIST_SAVE_PATH
fi

python split.py \
        --division=$DIVISION_NUM \
        --save-dir=$SPLIT_FILE_LIST_SAVE_PATH \
	--file-list=$FEATURE_FILE_LIST  \
	--file-prefix=$SPLIT_FILE_LIST_PREFIX  \
        &
       
wait


if [ ! -d $SAVE_DIR ];then
   mkdir SAVE_DIR
fi


#log
logPath="./logs"
if [ ! -d $logPath ];then
   mkdir  $logPath
fi
LOG_NAME="./logs/nohup-extract-log-megafce-challenge2"
imagelistprefix="/workspace/data/shiyongjie/code/batch_feature_extract/filelist/megaface2_aligned_face_filelist-split"
image_dir="/workspace/data/shiyongjie/code/tron_refinedet_gpu/megaface_aligned_face/aligned_imgs"

for ((i=0; i<$GPU_NUM; i++))
    do
        /bin/sleep 2
        nohup python extract_features_for_aligned_imagelist.py \
              --model=$MODEL_PATH \
              --image-list=$imagelistprefix${i}.txt \
              --image-dir=$image_dir  \
              --save-dir=$SAVE_DIR \
              --batch-size=20 \
              --image-size=3,112,112 \
              --add-flip \
              --gpu=$i \
              --save-format=.npy \
              --flip-sim > ${LOG_NAME}"_split_"${i}".txt" 2>&1 &
      echo "i="$i
    done
