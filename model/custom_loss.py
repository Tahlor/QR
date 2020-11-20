import matplotlib.pyplot as plt
from qr_decoder import zbar_decode
from create_qr_code2 import plot_qr
import numpy as np
from data_utils import gaussian_noise, change_contrast

def loss(img, qr_code):
    # what threshold for recognition?
    # take steps along sigmoid
    # (target/2)+.25-image
    pass

def random_color(gray):
    B = -1
    while not (0 <= B <= 255):
        R = np.random.randint(0, 255)
        G = np.random.randint(0, 255)
        B = int((gray - (R * 0.3 + G * 0.59)) / 0.11)
    return R, G, B

def colorize(img):
    new_image = np.zeros([img.shape[0],img.shape[1],3]).astype(np.uint)
    for i in range(img.shape[0]):
        for j in range(img.shape[0]):
            new_image[i,j] = random_color(img[i,j])
    return  new_image

def check_if_decodable(msg="this message"):
    img, qr = plot_qr(msg, plot=False)
    img = np.asarray(img)
    x = 250
    #img = 1-img
    img = (((img.astype(np.float) / x) + .5 - 1/(2*x))*255).astype(np.uint)

    # COLORIZE
    #img = colorize(img)

    # CONTRAST
    #img = change_contrast(img, 200)

    #img = gaussian_noise(img.astype(np.uint), .5)
    plt.imshow(img, cmap="gray", vmin=0, vmax=255); plt.show()
    print(img.max(), img.min())
    decoded_message = zbar_decode(img)
    return decoded_message


result = check_if_decodable()
print(result)