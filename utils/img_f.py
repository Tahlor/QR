import skimage
from skimage import io as io
from skimage import filters as filters
from skimage import transform as transform
import numpy as np

#These are all based on the OpenCV functions, to make the conversion to scikit image easier (also should make future changes easier as well)

def line(img,p1,p2,color,thickness=1,draw='set'):
    y1 = max(0,min(img.shape[0]-1,p1[1]))
    y2 = max(0,min(img.shape[0]-1,p2[1]))
    x1 = max(0,min(img.shape[1]-1,p1[0]))
    x2 = max(0,min(img.shape[1]-1,p2[0]))
    rr,cc = skimage.draw.line(y1,x1,y2,x2)

    if draw=='set':
        img[rr,cc]=color
    elif draw=='add':
        img[rr,cc]+=color
    elif draw=='mult':
        img[rr,cc]*=color
    if thickness>1:
        if x1<img.shape[1]-2 and y1<img.shape[0]-2 and x2<img.shape[1]-2 and y2<img.shape[0]-2:
            rr,cc = skimage.draw.line(y1+1,x1+1,y2+1,x2+1)
            if draw=='set':
                img[rr,cc]=color
            elif draw=='add':
                img[rr,cc]+=color
            elif draw=='mult':
                img[rr,cc]*=color
        if x1<img.shape[1]-2 and x2<img.shape[1]-2:
            rr,cc = skimage.draw.line(y1,x1+1,y2,x2+1)
            if draw=='set':
                img[rr,cc]=color
            elif draw=='add':
                img[rr,cc]+=color
            elif draw=='mult':
                img[rr,cc]*=color
        if y1<img.shape[0]-2 and y2<img.shape[0]-2:
            rr,cc = skimage.draw.line(y1+1,x1,y2+1,x2)
            if draw=='set':
                img[rr,cc]=color
            elif draw=='add':
                img[rr,cc]+=color
            elif draw=='mult':
                img[rr,cc]*=color
    if thickness>2:
        rr,cc = skimage.draw.line(y1-1,x1-1,y2-1,x2-1)
        if draw=='set':
            img[rr,cc]=color
        elif draw=='add':
            img[rr,cc]+=color
        elif draw=='mult':
            img[rr,cc]*=color
        rr,cc = skimage.draw.line(y1,x1-1,y2,x2-1)
        if draw=='set':
            img[rr,cc]=color
        elif draw=='add':
            img[rr,cc]+=color
        elif draw=='mult':
            img[rr,cc]*=color
        rr,cc = skimage.draw.line(y1-1,x1,y2-1,x2)
        if draw=='set':
            img[rr,cc]=color
        elif draw=='add':
            img[rr,cc]+=color
        elif draw=='mult':
            img[rr,cc]*=color
        if y1<img.shape[0]-2 and y2<img.shape[0]-2:
            rr,cc = skimage.draw.line(y1+1,x1-1,y2+1,x2-1)
            if draw=='set':
                img[rr,cc]=color
            elif draw=='add':
                img[rr,cc]+=color
            elif draw=='mult':
                img[rr,cc]*=color
        if x1<img.shape[1]-2 and x2<img.shape[1]-2:
            rr,cc = skimage.draw.line(y1-1,x1+1,y2-1,x2+1)
            if draw=='set':
                img[rr,cc]=color
            elif draw=='add':
                img[rr,cc]+=color
            elif draw=='mult':
                img[rr,cc]*=color
        assert(thickness<4)

def rectangle(img,c1,c2,color,thickness=1):
    line(img,c1,(c2[0],c1[1]),color,thickness)
    line(img,(c2[0],c1[1]),c2,color,thickness)
    line(img,c2,(c1[0],c2[1]),color,thickness)
    line(img,(c1[0],c2[1]),c1,color,thickness)

def imread(path,color=True):
    return io.imread(path,not color)

def imwrite(path,img):
    minV = img.min()
    maxV = img.max()
    if maxV>1 and minV>=0:
        img=img.astype(np.uint8)
    return io.imsave(path,img,plugin='pil')

def imshow(name,img):
    return io.imshow(img)

def show(): #replaces cv2.waitKey()
    return io.show()

def resize(img,dim,fx=None,fy=None,degree=3): #remove ",interpolation = cv2.INTER_CUBIC"
    #hasColor = len(img.shape)==3
    if dim[0]==0:
        downsize = fx<1 and fy<1
        
        return transform.rescale(img,(fy,fx),degree,anti_aliasing=downsize,preserve_range=True)
    else:
        downsize = dim[0]<img.shape[0] and dim[1]<img.shape[1]
        return transform.resize(img,dim,degree,anti_aliasing=downsize,preserve_range=True)

def otsuThreshold(img):
    #if len(img.shape)==3 and img.shape[2]==1:
    #    img=img[:,:,0]
    t = filters.threshold_otsu(img)
    return  t,(img>t)*255

def rgb2hsv(img):
    return skimage.color.rgb2hsv(img)
def hsv2rgb(img):
    return skimage.color.hsv2rgb(img)
def rgb2gray(img):
    return skimage.color.rgb2gray(img)
def gray2rgb(img):
    if len(img.shape) == 3:
        img=img[:,:,0]
    return skimage.color.gray2rgb(img)

def polylines(img,points,isClosed,color,thickness=1):
    if len(points.shape)==3:
        assert(points.shape[1]==1)
        points=points[:,0]
    if isClosed=='transparent':
        rr,cc = skimage.draw.polygon_perimeter(points[:,1],points[:,0],shape=img.shape)
        rr_f,cc_f = skimage.draw.polygon(points[:,1],points[:,0],shape=img.shape)
        img[rr_f,cc_f] = img[rr_f,cc_f]*0.7+np.array(color)*0.3
    elif isClosed:
        rr,cc = skimage.draw.polygon(points[:,1],points[:,0],shape=img.shape)
    else:
        rr,cc = skimage.draw.polygon_perimeter(points[:,1],points[:,0],shape=img.shape)
    img[rr,cc]=color

def warpAffine(img,M,shape=None):
    if shape is None:
        shape=img.shape
    if M.shape[0]==2: #OpenCV takes 2x3 instead of 3x3
        M = np.concatenate((M,np.array([[0.0,0.0,1.0]])),axis=0)
    T = transform.AffineTransform(M)
    return transform.warp(img,T,output_shape=shape)

def remap(img,map_x,map_y,interpolation=2,borderValue=None):
    return transform.warp(img,np.stack((map_y,map_x),axis=0),order=interpolation)

ROTATE_90_COUNTERCLOCKWISE=1
ROTATE_90_CLOCKWISE=3
ROTATE_180=2
def rotate(img,num_rot,degress=None):
    if num_rot is not None:
        return np.rot90(img,num_rot,axes=(0,1))
    else:
        raise NotImplementedError()

def getAffineTransform(src,dst):
    return transform.estimate_transform('affine',src,dst).params

def superimposeImages(img1, img2, img2_wt=None):
    """

    Args:
        img1 (nd-array): QR code
        img2 (nd-array): background image
        img2_wt:

    Returns:

    """
    if img1.ndim == 2:
        img1 = img1[:,:,np.newaxis]
    if img2_wt is None:
        img2_wt = np.clip(np.random.randn()/3+.2,0,.9)
    if isinstance(img2, str):
        img2 = np.array(cv2.imread(img2)).astype(np.uint8)*255
    #print(img1.shape, img1.dtype, img2.shape, img2.dtype)
    y,x,c = img1.shape
    #cropped = img2[:x,:y,:]
    img2 = resize(img2, (x,y))[:,:,np.newaxis]
    added_image = img2*img2_wt + img1*(1-img2_wt) #cv2.addWeighted(img2, img2_wt, img1, 1-img2_wt, 0)
    return added_image

if __name__ == "__main__":
    import sys
    input_image = sys.argv[1]
    img = imread(input_image)
    img = rotate(img,ROTATE_90_COUNTERCLOCKWISE)
    imshow('warped', img)
    show()
