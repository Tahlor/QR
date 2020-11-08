# QR Generation

## QR Decoder

Consider using a conditional GAN/VAE to be the QR generator?? You want something that, given a data message, 
it tells you the odds that the image is that message. The compressed encoding should be the actual message, maybe with some noise,
decoder trained to generate QR code from encoding. 