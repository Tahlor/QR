import cv2
import sys

class QRDecoder:
    def __init__(self):
        self.detector = cv2.QRCodeDetector()

    def decode(self, filename):
        """

        Args:
            filename (str or path):

        Returns:

        """
        # read the QRCODE image
        img = cv2.imread(str(filename))

        # detect and decode
        data, bbox, straight_qrcode = self.detector.detectAndDecode(img)

        # if there is a QR code
        if bbox is not None:
            print(f"QRCode data:\n{data}")
            # display the image with lines
            # length of bounding box
            n_lines = len(bbox)
            for i in range(n_lines):
                # draw all lines
                point1 = tuple(bbox[i][0])
                point2 = tuple(bbox[(i + 1) % n_lines][0])
                cv2.line(img, point1, point2, color=(255, 0, 0), thickness=2)
            return data
        else:
            print("No QR code found")
            return False


    def close(self):
        cv2.waitKey(0)
        cv2.destroyAllWindows()