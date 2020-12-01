import numpy as np
from scipy import ndimage
import random
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
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

    #assert image.ndim == 3
    shape = image.shape[:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]
    result = np.empty_like(image)
    if image.ndim ==3:
        for i in range(image.shape[2]):
            result[:, :, i] = map_coordinates(
                image[:, :, i], indices, order=spline_order, mode=mode).reshape(shape)
    elif image.ndim==2:
        result[:, :] = map_coordinates(
            image[:, :], indices, order=spline_order, mode=mode).reshape(shape)
    else:
        raise Exception("Should have 2 or 3 dimensions")
    return result
def occlude(img):
    # p=0.95
    # s_min=0.02
    # s_max=0.04
    # r_min=0.3
    # r_max = 1./r_min
    # size = 64
    # if np.random.uniform() > p:
    #     return tensor
    # while True:
    #     Se = np.random.uniform(s_min,s_max) * size * size
    #     re = np.random.uniform(r_min, r_max)
    #     He, We = np.sqrt(Se * re), np.sqrt(Se/re)
    #     xe, ye = np.random.uniform(0,size), np.random.uniform(0,size)
    #     if int(xe + We) <= size and int(ye + He) <= size:
    #         tensor[:, int(ye):int(ye + He), int(xe):int(xe + We)].fill(np.random.uniform()*2-1)
    #         return tensor
    w = np.random.randint(int(img.shape[0]/3))
    h = np.random.randint(int(img.shape[1]/3))
    x1 = np.random.randint(img.shape[0]-w)
    y1 = np.random.randint(img.shape[1]-h)
    img[x1:x1 + w, y1:y1+h] = np.random.uniform(0, 255, [w,h] if img.ndim==2 else [w,h,1])
    return img

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
