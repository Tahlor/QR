import cv2
import qr_decoder
from pyzbar.pyzbar import decode as pyzbar_decode
from pathlib import Path
import os
from collections import defaultdict

import qrcode
from qr_decoder_zoo.Python-QR-Decoder.QRMatrix import QRMatrix

def makeQR(text):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=11,
        border=2,
    )
    qr.add_data(text)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    return np.array(img)
    #img.save("tmp1.png")

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
    y, x, c = qr.shape
    cropped = background[:x, :y, :]
    added_image = cv2.addWeighted(cropped, threshold, qr, 1 - threshold, 0)
    #cv2.imwrite(str(output_file_path), added_image)
    return added_image #output_file_path

if __name__=='__main__':
    qr_decoders = {
            'pip install':qr_decoder.QRDecoder(),
            'opencv': cv.QRCodeDetector()
            'allenywang': None,
            'pyzbar': pyzbar_decode
            }
    qr_decode = {
            'pip install': lambda qr,a: qr.decode(a),
            'opencv': lambda qr,a: qr.detectAndDecode(a)[0]
            'allenywang': lambda qr,a: QRMatrix('decode',image=a).decode()
            'pyzbar': lambda qr,a: qr(a)[0].data
            }
    images = Path("./imagenet/images/dogs").rglob("*")
    num_images = len(images)
    test_strings=["short","medium 4io4\:][","long sdfjka349:fg,.<>fgok4t-={}.///gf"]


    for text in test_strings:
        results=defaultdict(lambda: 0)
        qr_image = makeQR(text)
        for f in images:
            background = cv2.imread(str(f))
            for name,qr in qr_decoders.items():
                qr_d = qr_decode[name]
                for p in range(20):
                    s = superimpose(background, qr_image, round(p*.05,2))
                    res = qr_d(qr,s)
                    print('{} {} : {}'.format(p*.05,name,res))
                    if res==text:
                        results[name]+=1
                        #s.rename(s.parent / (s.stem + "_DECODED" + s.suffix))
        print('For text: {} ========='.format(text))
        for name,count in results.items():
            print('{}\t{}/{}:\t{}'.format(name,count,num_images,count/num_images))
