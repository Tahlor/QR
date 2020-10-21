import qrcode
import sys

dest="test.png"
if len(sys.argv)>1:
    data=sys.argv[1]
    if len(sys.argv)>2:
        dest = sys.argv[2]
else:
    data='Some data'


qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=11,
    border=2,
)
qr.add_data(data)
qr.make(fit=True)

img = qr.make_image(fill_color="black", back_color="white")
img.save(dest)


# sudo apt-get install python-qrtools

#17 characters max
#qr = qrcode.QRCode(
#    version=1,
#    error_correction=qrcode.constants.ERROR_CORRECT_L,
#    box_size=11,
#    border=2,
#)

#14 characters max
#qr = qrcode.QRCode(
#    version=1,
#    error_correction=qrcode.constants.ERROR_CORRECT_M,
#    box_size=11,
#    border=2,
#)
