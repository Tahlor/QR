import cv2

background = cv2.imread('imagenet/images/dogs/2800688_afe83c164a.jpg')
qr = cv2.imread('test.png')
y,x,c = qr.shape
cropped = background[:x,:y,:]

image_wt = .2
added_image = cv2.addWeighted(cropped, image_wt, qr, 1-image_wt, 0)
cv2.imwrite('combined.png', added_image)
