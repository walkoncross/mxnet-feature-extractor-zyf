import os
import os.path as osp
import numpy as np
import argparse
import _init_paths
import sys
import json
from mxnet_feature_extractor import MxnetFeatureExtractor

def process_image_list(feat_extractor, img_list, 
                        image_dir=None, save_dir=None,save_type='.npy'):
    ftrs = feat_extractor.extract_features_for_image_list(img_list, image_dir)
#    np.save(osp.join(save_dir, save_name), ftrs)
    for i in range(len(img_list)):
        spl = osp.split(img_list[i])
        base_name = spl[1]
#        sub_dir = osp.split(spl[0])[1]
        sub_dir = spl[0]

        if sub_dir:
            save_sub_dir = osp.join(save_dir, sub_dir)
            if not osp.exists(save_sub_dir):
                os.makedirs(save_sub_dir)

        save_name = osp.splitext(base_name)[0] + save_type
        np.save(osp.join(save_sub_dir, save_name), ftrs[i])


def parse_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--image-list', type=str, help='image list file')
    parser.add_argument('--image-dir', type=str,
                        help='image root dir if image list contains relative paths')
    parser.add_argument('--save-dir', type=str, default='./rlt-features',
                        help='where to save the features')
    parser.add_argument('--batch-size', type=int, help='', default=80)
    parser.add_argument('--image-size', type=str,
                        help='', default='3,112,112')
    parser.add_argument('--add-flip', action='store_true',
                        help='use (oringal + flip)')
    parser.add_argument('--gpu', type=int, help='', default=0)
    parser.add_argument('--use-mean', action='store_true')
    parser.add_argument('--save-format', type=str, default='.npy',
                        help='either npy or bin, save output as: 1).npy format; 2).bin: megaface format.')
    parser.add_argument('--flip-sim', action='store_true',
                        help='To test similarity between original image and its flipped version.')

    parser.add_argument('--model', type=str, help='',
                        default='../model/model-r50-am-lfw/model,0')
    return parser.parse_args(argv)


def modify_config_json(config_json,args):
    str_json={}
    with open(config_json,'r') as fin:
        str_json=json.load(fin)
    fin.close()
    print 'before===>\n',str_json
    save_dir = args.save_dir
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    if "save_dir" in str_json:
        str_json["save_dir"]=save_dir
    gpu_id = args.gpu
    if "gpu_id" in str_json:
        str_json["gpu_id"]=gpu_id
    batch_size = args.batch_size

    if args.add_flip:
         if "mirror_trick" in str_json:
            str_json["mirror_trick"]=1
    if "batch_size" in str_json:
        str_json["batch_size"]=batch_size

    str_model= args.model
    if "network_model" in str_json:
        str_json["network_model"]=str_model
    print 'afer===>\n',str_json
    return str_json


def main(config_json_file, args):
    # if not osp.exists(save_dir):
    #     os.makedirs(save_dir)
    config_json=modify_config_json(config_json_file,args)
    image_list_file=args.image_list
    image_dir=args.image_dir
    save_dir=args.save_dir
    save_type=args.save_format
    if not os.path.isfile(image_list_file):
        print "image_list_file does not exist, exit..."
        exit
    if not os.path.isdir(save_dir):
        print save_dir," does not exist,create..."
        os.makedirs(save_dir)

    fp = open(image_list_file,'r')
    # init a feat_extractor
    print '\n===> init a feat_extractor'
    feat_extractor = MxnetFeatureExtractor(config_json)
    print('===> feat_extractor configs:\n', feat_extractor.config)

    batch_size = feat_extractor.get_batch_size()
    print 'feat_extractor can process %d images in a batch' % batch_size

    img_list = []
    cnt = 0
    batch_cnt = 0

    for line in fp:
        if line.startswith('#'):
            continue

        items = line.split()
        img_list.append(items[0].strip())
        cnt += 1

        if cnt == batch_size:
            batch_cnt += 1
            print '\n===> Processing batch #%d with %d images' % (batch_cnt, cnt)

            process_image_list(feat_extractor, img_list, image_dir, save_dir,save_type)
            cnt = 0
            img_list = []

    if cnt > 0:
        batch_cnt += 1
        print '\n===> Processing batch #%d with %d images' % (batch_cnt, cnt)
        process_image_list(feat_extractor, img_list, image_dir, save_dir,save_type)

    fp.close()


if __name__ == '__main__':
    config_json = './extractor_config.json'
    argv=parse_args(sys.argv[1:])
    print 'argv---->\n',argv
    main(config_json,argv)

    # modify_config_json(config_json,argv)

    # save_dir = 'rlt_features'
    # image_dir = r'../../test_data/face_chips_112x112'
    # image_list_file = r'../../test_data/face_chips_list.txt'
    # main(config_json, save_dir, image_list_file, image_dir)
