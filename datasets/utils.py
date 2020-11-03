from scipy import ndimage
import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import torch
from easydict import EasyDict as edict
import warnings
import string

def is_iterable(object, string_is_iterable=True):
    """Returns whether object is an iterable. Strings are considered iterables by default.

    Args:
        object (?): An object of unknown type
        string_is_iterable (bool): True (default) means strings will be treated as iterables
    Returns:
        bool: Whether object is an iterable

    """

    if not string_is_iterable and type(object) == type(""):
        return False
    try:
        iter(object)
    except TypeError as te:
        return False
    return True

def compare_lists(a,b):
    if not is_iterable(a):
        return a==b
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i]!=b[i]:
            return False
    return True

def create_alphabet(alphabet_description):
    if alphabet_description == "printable":
        return string.printable # 101
    elif alphabet_description == "digits":
        return string.digits  # 11
    elif alphabet_description == "alhanumeric":
        return string.ascii_letters + string.digits # 63


def make_consistent(d1, d1_key, d2, d2_key):
    if d1_key not in d1:
        d1[d1_key] = d2[d2_key]
    elif d2_key not in d2:
        d2[d2_key] = d1[d1_key]
    else:
        if not compare_lists(d1[d1_key], d2[d2_key]):
            warnings.warn(f"{d1_key} is {d1[d1_key]} whereas {d2_key} is {d2[d2_key]}; using {d1_key}")
            d2[d2_key] = d1[d1_key]

def make_config_consistent(config):
    config = edict(config)
    #make_consistent(config.data_loader, "image_size", config.model, "input_size")
    
    # max_message_len
    # choose alphabet / num_char_class
    if "alphabet_description" in config.data_loader:
        if "alphabet" in config.data_loader:
            warnings.warn("Alphabet and alphabet description specified, using alphabet_description")
        config.data_loader.alphabet = create_alphabet(config.data_loader.alphabet_description)

    alphabet_length = len(set(config.data_loader.alphabet)) + 1
    if "num_char_class" in config.model and config.model.num_char_class != alphabet_length:
        warnings.warn(f"num_char_class incorrect, using {alphabet_length}")
        config.model.num_char_class = alphabet_length
    print(config.model)
    print(config.data_loader)
    return config

def img2tensor(img):
    return (torch.from_numpy(np.asarray(img))[None, ...].float() / 255) * 2 - 1

def blur(img, max_intensity=1.5):
    max_intensity = np.random.uniform(0, max_intensity)
    return ndimage.gaussian_filter(img, max_intensity)

def gaussian_noise(img, max_intensity=10, logger=None):
    """
        Expects images on 0-255 scale
        max_intensity: .1 - light haze, 1 heavy

        Adds random noise to image
    """

    random_state = np.random.RandomState()
    sd = np.random.rand() * max_intensity / 2
    # min(abs(np.random.normal()) * max_intensity / 2, max_intensity / 2)
    #sd = max_intensity / 2  # ~95% of observations will be less extreme; if max_intensity=1, we set so 95% of multipliers are <1
    noise_mask = random_state.randn(*img.shape, ) * sd  # * 2 - max_intensity # min -occlusion, max occlusion
    noise_mask = np.clip(noise_mask, -1, 1) * 255/2
    noisy_img = np.clip(img + noise_mask, 0, 255)
    return noisy_img.astype(int)

def superimpose_images(img1, img2, img2_wt=.5):
    """

    Args:
        img1 (nd-array):
        img2 (nd-array):
        img2_wt:

    Returns:

    """
    if isinstance(img2, str):
        img2 = cv2.imread(img2)

    y,x,c = img1.shape
    #cropped = img2[:x,:y,:]
    img2 = cv2.resize(img2, (x,y))
    added_image = cv2.addWeighted(img2, 1-img2_wt, img1, img2_wt, 0)
    return added_image


def homography(image):
    h = image.shape[0]
    w = image.shape[1]

    new_top_left = np.random.randint(0, int(h * .8)), np.random.randint(0, int(w * .8))
    new_top_right = np.random.randint(0, int(h * .8)), w - np.random.randint(0, int(w * .8))
    new_bottom_right = h - np.random.randint(0, int(h * .8)), w - np.random.randint(0, int(w * .8))
    new_bottom_left = h - np.random.randint(0, int(h * .8)), np.random.randint(0, int(w * .8))

    src = np.array([[0, 0], [0, w], [h, w], [h, 0]])
    dest = np.array([new_top_left, new_top_right, new_bottom_right, new_bottom_left])
    homography, status = cv2.findHomography(src, dest)

    # im_out = cv2.warpPerspective(image, h, (im_dst.shape[1], im_dst.shape[0]))
    image = cv2.warpPerspective(image, homography, (w, h))
    return image

def elastic_transform(image, alpha=2.5, sigma=.5, random_state=None, spline_order=1, mode='nearest'):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    alpha = np.random.uniform(1,alpha)

    if random_state is None:
        random_state = np.random.RandomState(None)

    assert image.ndim == 3
    shape = image.shape[:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]
    result = np.empty_like(image)
    for i in range(image.shape[2]):
        result[:, :, i] = map_coordinates(
            image[:, :, i], indices, order=spline_order, mode=mode).reshape(shape)
    return result

