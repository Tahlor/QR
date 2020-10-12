import cv2
import qr_decoder
from pathlib import Path
import os

def superimpose(background_img, qr_img, threshold, output_folder="images/superimposed"):
    """

    Args:
        background_img (str OR Path): the path
        qr_img (str): the path
        threshold: between 0 and 1
        output_folder (str): the path

    Returns:

    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True,exist_ok=True)
    background_img = Path(background_img)
    output_file_path = output_folder / (background_img.stem + str(threshold) + background_img.suffix)

    background = cv2.imread(str(background_img))
    qr = cv2.imread(str(qr_img))
    y, x, c = qr.shape
    cropped = background[:x, :y, :]
    added_image = cv2.addWeighted(cropped, threshold, qr, 1 - threshold, 0)
    cv2.imwrite(str(output_file_path), added_image)
    return output_file_path

if __name__=='__main__':
    qr = qr_decoder.QRDecoder()

    for f in Path("./imagenet/images/dogs").rglob("*"):
        for p in range(20):
            s = superimpose(f, "test.png", round(p*.05,2))
            if qr.decode(s):
                s.rename(s.parent / (s.stem + "_DECODED" + s.suffix))
