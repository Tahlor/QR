#from utils import util
#from test_qr_decoders import superimpose
from skimage import io as io
import matplotlib.pyplot as plt
from skimage.filters import gaussian, unsharp_mask
from skimage.exposure import adjust_log

def imread(path,color=True):
    return io.imread(path,not color)

def zbar_decode(img):
    from pyzbar.pyzbar import decode as pyzbar_decode
    #img = img.mean(axis=2)
    res=pyzbar_decode(img)

    if len(res)==0:
        if len(img.shape)>2:
            imgg = img.mean(axis=2)
            res=pyzbar_decode(imgg)
        if len(res)==0:
            imgb = gaussian(img,1,preserve_range=True)
            res=pyzbar_decode(imgb)
            if len(res)==0:
                imgc = adjust_log(img)
                res=pyzbar_decode(imgc)
            #if len(res)==0:
            #    imgs = unsharp_mask(img,preserve_range=True)
            #    res=pyzbar_decode(imgs)
    if len(res)==0:
        return None
    elif len(res)==1:
        return res[0].data.decode("utf-8")
    else:
        return [r.data.decode("utf-8") for r in res]
    # polygon = {list: 4} [Point(x=1305, y=502), Point(x=1536, y=501), Point(x=1533, y=274), Point(x=1306, y=274)]
    # rect = {Rect: 4} Rect(left=1305, top=274, width=231, height=228)

def add_corners():
    pass

def fade_in_mask(art, qr_img, qr_intensity):
    """

    Args:
        art:
        qr_img: 0's are ignored; -1/1 are faded in

    Returns:

    """
    faded_in = art.copy()
    faded_in[faded_in !=0] = ((1-qr_intensity)*art[faded_in !=0] + qr_intensity*qr_img[faded_in !=0])
    return faded_in

def main(qr_img, art):
    qr_img = imread(str(qr_img))
    art = imread(str(art))

    for i in range(0,11):
        output_image = fade_in_mask(art=art, qr_img=qr_img, qr_intensity=i/10.)
        result = zbar_decode(output_image)
        if result:
            print(f"Decoded at {i*10}%")
            break
    # save the output image
    plt.imshow(output_image); plt.show()

if __name__=='__main__':

    qr_img = "../images/post_process/gen_gt_75156.png"
    art = qr_img.replace("gen_gt", "gen_samples")# "./images/post_process/gen_samples_75156.png"
    main(qr_img, art)