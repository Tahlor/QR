#import cv2
#import qr_decoder
from pyzbar.pyzbar import decode as pyzbar_decode
from pathlib import Path
import os
from collections import defaultdict
import numpy as np
import utils.img_f as cv2

import qrcode
#from qr_decoder_zoo.Python_QR_Decoder.QRMatrix import QRMatrix

def zbar_decode(img):
    res=pyzbar_decode(img)
    if len(res)==0:
        return None
    elif len(res)==1:
        return res[0].data.decode("utf-8")
    else:
        return [r.data.decode("utf-8") for r in res]

def makeQR(text,size=None):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=11,
        border=2,
    )
    qr.add_data(text)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    img = np.array(img)[:,:,None].repeat(3,axis=2).astype(np.uint8)*255
    if size is not None:
        img = cv2.resize(img,(size,size),degree=0)
    return img
    #img.save("tmp1.png")

def contrast(qr_img,amount):
    qr_img = np.copy(qr_img)
    qr_img=qr_img*(1-amount)
    qr_img+=amount/2
    return qr_img

def superimpose(background_img, qr_img, threshold, output_folder="images/superimposed"):
    """

    Args:
        background_img (str OR Path): the path
        qr_img (str): the path
        threshold: between 0 and 1
        output_folder (str): the path

    Returns:

    """
    #background = cv2.imread(str(background_img))
    #qr = cv2.imread(str(qr_img))
    y, x, c = qr_img.shape
    cropped = background_img[:x, :y, :]
    padx = max(0,x-cropped.shape[1])
    pady = max(0,y-cropped.shape[0])
    if padx>0 or pady>0:
        cropped = np.pad(cropped,((pady//2,pady//2+pady%2),(padx//2,padx//2+padx%2),(0,0)))
    #added_image =  cv2.addWeighted(cropped, threshold, qr_img, 1 - threshold, 0)
    added_image = cropped*threshold + qr_img*(1 - threshold)
    #cv2.imwrite(str(output_file_path), added_image)
    return added_image #output_file_path

if __name__=='__main__':
    qr_decoders = {
            #'opencv': cv2.QRCodeDetector(),
            #'allenywang': None,
            'pyzbar': None
            }
    qr_decode = {
            #'opencv': lambda qr,a: qr.detectAndDecode(a)[0],
            #:'allenywang': lambda qr,a: QRMatrix('decode',image=a).decode(),
            'pyzbar': lambda qr,a: zbar_decode(a)
            }
    images = list(Path("./imagenet/images/dogs").rglob("*"))
    num_images = len(images)
    test_strings=["short","medium 4io4\:][","long sdfjka349:fg,.<>fgok4t-={}.///gf"]

    

    for text in test_strings:
        results=defaultdict(lambda: 0)
        avg_max_interpolation = defaultdict(list)
        qr_image = makeQR(text,256)
        for imN,f in enumerate(images):
            background = cv2.imread(str(f))
            background = cv2.resize(background,(256,256))
            for name,qr in qr_decoders.items():
                qr_d = qr_decode[name]

                max_hit=0
                for p in range(20):
                    mix = round(p*.05,2)
                    #s = superimpose(background, qr_image, mix)
                    s = contrast(qr_image,round(p*.05,2))
                    cv2.imwrite('tmp.png',s)
                    s_ = cv2.imread('tmp.png')
                    res = qr_d(qr,s_)
                    #if mix>0.5:
                    #    import pdb;pdb.set_trace()
                    #print('{} {} : {}'.format(p*.05,name,res))
                    if res==text:
                        results[name]+=1
                        #s.rename(s.parent / (s.stem + "_DECODED" + s.suffix))
                        max_hit = max(max_hit,mix)
                        cv2.imwrite('got_this2/cont{}_{}_{}.png'.format(len(text),imN,p),s)
                avg_max_interpolation[name].append(max_hit)
        print('For text: {} ========='.format(text))
        for name,count in results.items():
            print('{}\t{}/{}:\t{}\thighest interpolation: {}'.format(name,count,num_images*20,count/(20*num_images),np.mean(avg_max_interpolation[name])))

