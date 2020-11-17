from utils.util import zbar_decode
import utils.img_f as img_f
import sys
import numpy as np
from skimage.filters import gaussian

paths = sys.argv[1:]
#path = sys.argv[1]
success=[]
failure=[]
for path in paths:
    img = img_f.imread(path)
    #print(img.shape)
    #print('min: {}, max: {}'.format(img.min(),img.max()))

    #print('one image')
    res=zbar_decode(img)
    print(res)
    #print('num: {}'.format(len(res)))

    if res is not None:
        success.append(path)
    else:
        failure.append(path)

    if img.shape[0]>600:
        imgs=[]
        for r in range(4):
            for c in range(8):
                imgs.append(img[2+(256+2)*r:2+(256+2)*r+256,2+(256+2)*c:2+(256+2)*c+256])
        #imgs = [np.pad(img,((20,20),(20,20),(0,0)),constant_values=255)]

        print('individually')
        nres=[]
        for img in imgs:
            #m = img.mean()
            #img = np.pad(img,((10,10),(10,10),(0,0)),constant_values=255)
            res=zbar_decode(img)
            #import pdb;pdb.set_trace()
            if res is not None:
                nres.append(res)
            #print(res)
            #img_f.imshow('x',img)
            #img_f.show()
        #print(nres)
        print('num: {}'.format(len(nres)))

print('Failures:')
print(failure)
print('Success {}/{}'.format(len(success),len(paths)))

