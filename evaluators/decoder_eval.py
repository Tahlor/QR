#from skimage import color, io
import os
import numpy as np
import torch
from utils import util
import math
from collections import defaultdict
from trainer import *
import utils.img_f as img_f

def AdvancedQRDataset2_eval(config,instance, trainer, metrics, outDir=None, startIndex=None, lossFunc=None, toEval=None):
    return Decoder_eval(config,instance, trainer, metrics, outDir, startIndex, lossFunc, toEval)
def Decoder_eval(config,instance, trainer, metrics, outDir=None, startIndex=None, lossFunc=None, toEval=None):
    losses,log,chars = trainer.run(instance)
    images = instance['image'].numpy()
    gt = instance['gt_char']
    batchSize = len(gt)
    
    for b in range(batchSize):
        print('{}   GT: {}'.format(b,gt[b]))
        print('{} pred: {}'.format(b,chars[b]))
        if outDir is not None:
            save_path = os.path.join(outDir,'{}_G:{}_P:{}.png'.format(startIndex+b,gt[b],chars[b]))
            img = (images[b,0]+1)/2
            img_f.imwrite(save_path,img)

    for name,v in losses.items():
        log['name']=v.item()
    return (
             log,
             log
            )



