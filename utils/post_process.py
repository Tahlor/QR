from utils import util
from test_qr_decoders import superimpose
import utils.img_f as cv2

def add_corners():
    pass

def fade_in_mask(art, qr_img, qr_intensity):
    """

    Args:
        art:
        qr_img: 0's are ignored

    Returns:

    """

    pass

def main():

    qr_img = "./images/post_process/gen_gt_75156.png"
    art =    qr_img.replace("gen_gt", "gen_samples")# "./images/post_process/gen_samples_75156.png"
    qr_img = cv2.imread(str(qr_img))
    art = cv2.imread(str(art))


    for i in range(0,10):
        output_image = fade_in_mask(art=art, qr_img=qr_img)
        result = util.zbar_decode(output_image)
        if result:
            break
    # save the output image

if __name__=='__main__':
    pass