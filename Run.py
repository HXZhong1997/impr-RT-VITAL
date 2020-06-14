import os
from os.path import join, isdir
from tracker import *
import numpy as np

import argparse

import pickle

import math


def genConfig(seq_path, set_type):

    path, seqname = os.path.split(seq_path)


    if set_type == 'OTB':
        ############################################  have to refine #############################################
        if (seqname == 'Jogging.1') or (seqname == 'Jogging.2'):
            img_list = sorted([os.path.join(path,'Jogging') + '/img/' + p for p in os.listdir(os.path.join(path,'Jogging') + '/img') if os.path.splitext(p)[1] == '.jpg'])
        elif seqname == ('Skating2.1') or (seqname == 'Skating2.2'):
            img_list = sorted([os.path.join(path,'Skating2') + '/img/' + p for p in os.listdir(os.path.join(path,'Skating2') + '/img') if os.path.splitext(p)[1] == '.jpg'])
        elif seqname == 'Human4.2':
            img_list = sorted([os.path.join(path,'Human4')+'/img/' + p for p in os.listdir(os.path.join(path,'Human4') + '/img') if os.path.splitext(p)[1] == '.jpg'])

        else:
            img_list = sorted([seq_path + '/img/' + p for p in os.listdir(seq_path + '/img') if os.path.splitext(p)[1] == '.jpg'])

        if seqname == 'Jogging.1':
            gt = np.loadtxt(os.path.join(path,'Jogging') + '/groundtruth_rect.1.txt')
        elif seqname == 'Jogging.2':
            gt = np.loadtxt(os.path.join(path,'Jogging') + '/groundtruth_rect.2.txt')
        elif seqname == 'Skating2.1':
            gt = np.loadtxt(os.path.join(path,'Skating2') + '/groundtruth_rect.1.txt')
        elif seqname == 'Skating2.2':
            gt = np.loadtxt(os.path.join(path, 'Skating2') + '/groundtruth_rect.2.txt')
        elif seqname =='Human4.2':
            gt = np.loadtxt(os.path.join(path,'Human4') + '/groundtruth_rect.2.txt', delimiter=',')
        elif (seqname=='Jogging') or (seqname == 'Rubik') or (seqname == 'Singer1') or (seqname == 'Subway') \
                or (seqname == 'Surfer') or (seqname == 'Sylvester') or (seqname == 'Toy') or (seqname == 'Twinnings') \
                or (seqname == 'Vase') or (seqname == 'Walking') or (seqname == 'Walking2') or (seqname == 'Woman')   :
            gt = np.loadtxt(seq_path + '/groundtruth_rect.txt')
        else:
            gt = np.loadtxt(seq_path + '/groundtruth_rect.txt', delimiter=',')

        if seqname == 'David':
            img_list = img_list[299:]
          
        if seqname == 'Football1':
            img_list = img_list[0:74]
        if seqname == 'Freeman3':
            img_list = img_list[0:460]
        if seqname == 'Freeman4':
            img_list = img_list[0:283]
        if seqname == 'Diving':
            img_list = img_list[0:215]
        # if seqname == 'Tiger1':
        #    img_list = img_list[5:]


    if set_type == 'GOT-10k':
        img_list = sorted([seq_path +'/' + p for p in os.listdir(seq_path) if os.path.splitext(p)[1] == '.jpg']);
        gt = np.loadtxt(seq_path + '/groundtruth.txt', delimiter=',')
        gt = gt.reshape(1,4)

        ##polygon to rect
    if gt.shape[1] == 8:
        x_min = np.min(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_min = np.min(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        x_max = np.max(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_max = np.max(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)

    return img_list, gt


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-set_type", default = 'OTB' )
    parser.add_argument("-model_path", default = './models/rt-mdnet.pth')
    parser.add_argument("-result_path", default = './result/results.dict')
    parser.add_argument("-visual_log",default=False, action= 'store_true')
    parser.add_argument("-visualize",default=False, action='store_true')
    parser.add_argument("-adaptive_align",default=True, action='store_false')
    parser.add_argument("-padding",default=1.2, type = float)
    parser.add_argument("-jitter",default=True, action='store_false')
    parser.add_argument("-num2drop",default=288, type = int)
    parser.add_argument("-g_lr",default=0.003,type=float)

    args = parser.parse_args()

    ##################################################################################
    #########################Just modify opts in this script.#########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################
    ## option setting
    opts['model_path']=args.model_path
    opts['result_path']=args.result_path
    opts['visual_log']=args.visual_log
    opts['set_type']=args.set_type
    opts['visualize'] = args.visualize
    opts['adaptive_align'] = args.adaptive_align
    opts['padding'] = args.padding
    opts['jitter'] = args.jitter
    opts['num2drop'] = args.num2drop
    opts['g_lr']=args.g_lr
    ##################################################################################
    ############################Do not modify opts anymore.###########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################
    print opts


    ## path initialization
    dataset_path = '/data/zhonghaoxiang/dataset/'

    seq_home = dataset_path + opts['set_type']
    # seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
    with open('./OTB100.txt') as f:
        seq_list = f.read()
    seq_list = seq_list.split()
    #with open('/data/zhonghaoxiang/dataset/OTB100.txt') as f:
    #    seq_list = f.read()
    # seq_list = seq_list.split()
    #process seq_list for OTB dataset
    '''
    if opts['set_type']=='OTB':
        jogging_idx = seq_list.index('Jogging')
        seq_list.insert(jogging_idx,'Jogging.2')
        seq_list.insert(jogging_idx, 'Jogging.1')
        skating2_idx=seq_list.index('Skating2')
        seq_list.insert(skating2_idx,'Skating2.2')
        seq_list.insert(skating2_idx, 'Skating2.1')
        seq_list.remove('Jogging')
        seq_list.remove('Skating2')
    '''
    iou_list=[]
    fps_list=dict()
    spf_list=dict()
    bb_result = dict()
    result = dict()

    iou_list_nobb=[]
    bb_result_nobb = dict()
    num2drop=opts['num2drop']
    g_lr=opts['g_lr']
    g_lr_upd=opts['g_lr']
    
    for num, seq in enumerate(seq_list):
        if num < -1:
            continue
        
        seq_path = seq_home + '/' + seq
        img_list, gt = genConfig(seq_path, opts['set_type'])

        
        result_bb, fps, result_nobb, spf = run_mdnet(img_list, gt[0], num2drop, g_lr, g_lr_upd, gt, seq=seq, display=opts['visualize'])
        enable_frameNum = 0.
                    
        bb_result[seq] = result_bb
        fps_list[seq] = fps
        spf_list[seq] = spf
        
        bb_result_nobb[seq] = result_nobb
        
        print '{} {} : fps:{}'.format(num, seq, sum(fps_list.values()) / len(fps_list))
        
    result['bb_result'] = bb_result
    result['fps'] = fps_list
    result['bb_result_nobb'] = bb_result_nobb
    result['spf'] = spf_list
    result_path = opts['result_path']
    with open(result_path, 'wb') as f:
        pickle.dump(result, f)
    print('*** Result saved at \'{}\'. Run ./results/save_txt.py to decode as txt files.'.format(opts['result_path']))

