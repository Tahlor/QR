from datasets import image_and_qr_dataset
import math
import sys, os
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Polygon
import numpy as np
import torch
import utils.img_f as img_f

saveHere=None
linenum=0

def display(data):
    global saveHere, linenum
    gts=[]
    batchSize = data['image'].size(0)
    for b in range(batchSize):
        img = (data['image'][b].permute(1,2,0)+1)/2.0
        qr_img = (data['qr_image'][b].permute(1,2,0)+1)/2.0
        gt = data['gt_char'][b]
        gts.append(gt)
        #print(label[:data['label_lengths'][b],b])
        print(gt)

        #cv2.imshow('line',img.numpy())
        #cv2.waitKey()

        #fig = plt.figure()

        #ax_im = plt.subplot()
        #ax_im.set_axis_off()
        #if img.shape[2]==1:
        #    ax_im.imshow(img[0])
        #else:
        #    ax_im.imshow(img)

        #plt.show()
        #if saveHere is not None:
        #    cv2.imwrite(os.path.join(saveHere,'{}.png').format(linenum),img.numpy()*255)
        #    linenum+=1
        img_f.imshow('image',img.numpy())

        img_f.show()
        img_f.imshow('qr image',qr_img[...,0].numpy())
        img_f.show()
        
    #print('batch complete')
    return gts


if __name__ == "__main__":
    dirPath = sys.argv[1]
    if len(sys.argv)>2:
        start = int(sys.argv[2])
    else:
        start=0
    if len(sys.argv)>3:
        repeat = int(sys.argv[3])
    else:
        repeat=1
    data=image_and_qr_dataset.ImageAndQRDataset(dirPath=dirPath,split='train',config={
        'QR_dataset':{
            'data_set_name': 'SimpleQRDataset',
		"final_size": 256,
                "total_random": 17
            },
        #'image_dataset_name': 'LSUN',
        #'image_class': 'bedroom'
        'image_dataset_name': 'simple',
        'image_dataset_config': {'size':256}
})
    #data.cluster(start,repeat,'anchors_rot_{}.json')

    dataLoader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True, num_workers=0, collate_fn=image_and_qr_dataset.collate)
    dataLoaderIter = iter(dataLoader)

        #if start==0:
        #display(data[0])
    for i in range(0,start):
        print(i)
        dataLoaderIter.next()
        #display(data[i])
    gts=[]
    try:
        while True:
            #print('?')
            gts+=display(dataLoaderIter.next())
    except StopIteration:
        print('done')

    with open(os.path.join(dirPath,'test_gt.txt'),'w') as out:
        for gt in gts:
            out.write(gt+'\n')
