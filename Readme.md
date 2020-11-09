# QR Generation

## QR Decoder

Consider using a conditional GAN/VAE to be the QR generator?? You want something that, given a data message, 
it tells you the odds that the image is that message. The compressed encoding should be the actual message, maybe with some noise,
decoder trained to generate QR code from encoding. 

Pass in message, use VAE style decoder; in GAN loss, use NLL. Train (all) models
by 1) using as much noise/occlusion as possible while still being valid. 

What you really want to do is occlude some parts, then determine the
probability you can occlude a certain part with it still being readable. 
The model needs to have some distribution associated with it, so 
that we turn a knob and the most important parts of the QR code are fixed 
first, followed by the less essential bits. OR rather, we figure out
the threshold the scanner needs to recognize a block, change saturates at this point.

Or just make a predictor, given the true image and a corrupted one,
what is the smallest change to make it readable. RL or heuristic 
search problem.
