import os
import json
import logging
import argparse
import torch
from model import *
from model.metric import *
from model.loss import *
from logger import Logger
from trainer import *
from data_loader import getDataLoader
from evaluators import *
import math
from collections import defaultdict
import pickle, csv
import qrcode
from utils import img_f



def main(resume,saveDir,numberOfImages,message,qr_size,qr_border,qr_version,gpu=None, config=None,  addToConfig=None):
    if resume is not None:
        checkpoint = torch.load(resume, map_location=lambda storage, location: storage)
        print('loaded iteration {}'.format(checkpoint['iteration']))
        loaded_iteration = checkpoint['iteration']
        if config is None:
            config = checkpoint['config']
        else:
            config = json.load(open(config))
        for key in config.keys():
            if type(config[key]) is dict:
                for key2 in config[key].keys():
                    if key2.startswith('pretrained'):
                        config[key][key2]=None
    else:
        checkpoint = None
        config = json.load(open(config))
        loaded_iteration = None
    config['optimizer_type']="none"
    config['trainer']['use_learning_schedule']=False
    config['trainer']['swa']=False
    if gpu is None:
        config['cuda']=False
    else:
        config['cuda']=True
        config['gpu']=gpu
    addDATASET=False
    if addToConfig is not None:
        for add in addToConfig:
            addTo=config
            printM='added config['
            for i in range(len(add)-2):
                addTo = addTo[add[i]]
                printM+=add[i]+']['
            value = add[-1]
            if value=="":
                value=None
            elif value[0]=='[' and value[-1]==']':
                value = value[1:-1].split('-')
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            addTo[add[-2]] = value
            printM+=add[-2]+']={}'.format(value)
            print(printM)
            if (add[-2]=='useDetections' or add[-2]=='useDetect') and value!='gt':
                addDATASET=True

        
    #config['data_loader']['batch_size']=math.ceil(config['data_loader']['batch_size']/2)
    else:
        vBatchSize = batchSize

    if checkpoint is not None:
        if 'state_dict' in checkpoint:
            model = eval(config['model']['arch'])(config['model'])
            keys = list(checkpoint['state_dict'].keys())
            my_state = model.state_dict()
            my_keys = list(my_state.keys())
            for mkey in my_keys:
                if mkey not in keys:
                    checkpoint['state_dict'][mkey]=my_state[mkey]
                #else:
                #    print('{} me: {}, load: {}'.format(mkey,my_state[mkey].size(),checkpoint['state_dict'][mkey].size()))
            for ckey in keys:
                if ckey not in my_keys:
                    del checkpoint['state_dict'][ckey]
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model = checkpoint['model']
    else:
        model = eval(config['arch'])(config['model'])
    model.eval()

    #generate normal QR code
    qr = qrcode.QRCode(
            version=qr_version,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=1,
            border=qr_border,
            mask_pattern=None#self.mask_pattern,
        )
    qr.add_data(message)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    qr_img = np.array(qr_img)
    qr_img = img_f.resize(qr_img,(qr_size,qr_size),degree=0)
    if qr_img.max()==255:
        qr_img=qr_img/255
    qr_img = qr_img*2 -1

    qr_img=torch.from_numpy(qr_img[None,None,...]).float() #add batch and color
    qr_img=qr_img.expand(numberOfImages,-1,-1,-1)


    with torch.no_grad():
        if gpu is not None:
            qr_img = qr_img.to(gpu)
        gen_image = model(qr_img)
        gen_image = gen_image.clamp(-1,1)
        gen_image = (gen_image+1)/2
        gen_image=gen_image.cpu().permute(0,2,3,1)
        for b in range(numberOfImages):
            path = os.path.join(saveDir,'{}.png'.format(b))
            img_f.imwrite(path,gen_image[b])


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Evaluator/Displayer')
    parser.add_argument('-c', '--checkpoint', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--savedir', default=None, type=str,
                        help='path to directory to save result images (default: None)')
    parser.add_argument('-n', '--number', default=1, type=int,
                        help='number of generations to produce (default 1)')
    parser.add_argument('-g', '--gpu', default=None, type=int,
                        help='gpu number (default: cpu only)')
    parser.add_argument('-f', '--config', default=None, type=str,
                        help='config override')
    parser.add_argument('-a', '--addtoconfig', default=None, type=str,
                        help='Arbitrary key-value pairs to add to config of the form "k1=v1,k2=v2,...kn=vn".  You can nest keys with k1=k2=k3=v')
    parser.add_argument('-m', '--message', default=None, type=str,
            help='The message to put into the QR code (default 2)')
    parser.add_argument('-b', '--border', default=2, type=int,
                        help='padding around QR code')
    parser.add_argument('-s', '--size', default=256, type=int,
                        help='size of QR code (pixels) default 256')
    parser.add_argument('-v', '--version', default=1, type=int,
                        help='QR code version (default 1)')
    #parser.add_argument('-E', '--special_eval', default=None, type=str,
    #                    help='what to evaluate (print)')

    args = parser.parse_args()

    addtoconfig=[]
    if args.addtoconfig is not None:
        split = args.addtoconfig.split(',')
        for kv in split:
            split2=kv.split('=')
            addtoconfig.append(split2)

    config = None
    if args.checkpoint is None and args.config is None:
        print('Must provide checkpoint (with -c)')
        exit()
    if args.message is None:
        print('Must provide message (-m)')
        exit()

    if args.gpu is not None:
        with torch.cuda.device(args.gpu):
            main(args.checkpoint, args.savedir, args.number, args.message, args.size, args.border, args.version, gpu=args.gpu, config=args.config, addToConfig=addtoconfig)
    else:
        main(args.checkpoint, args.savedir, args.number, args.message,args.size, args.border, args.version, gpu=args.gpu, config=args.config, addToConfig=addtoconfig)
