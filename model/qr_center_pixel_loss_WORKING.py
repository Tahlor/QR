import matplotlib.pyplot as plt

import torch
from torch import nn
import numpy as np
from scipy import ndimage

class QRCenterPixelLoss(nn.Module):
    def __init__(self, img_size, qr_size, padding, threshold=0, bigger=False, split=False, factor=1.0,
                 no_corners=False, blur=False):
        super(QRCenterPixelLoss, self).__init__()
        assert (qr_size <= 33)
        self.threshold = threshold
        self.split = split
        self.no_corners = no_corners
        # create weight mask
        self.mask = torch.FloatTensor(img_size, img_size).zero_()
        cell_size = img_size / (qr_size + 2 * padding)

        if no_corners:
            split = self.split = False  # hack; turn off split, don't mask corners below
        else:
            if split:
                mask_corners = self.mask_corners = torch.FloatTensor(img_size, img_size).zero_()
            else:
                mask_corners = self.mask

            # mask achors (with padding)
            top_left_left_x = round((padding - 1) * cell_size)
            top_left_right_x = round((padding + 7 + 1) * cell_size)
            top_left_top_y = round((padding - 1) * cell_size)
            top_left_bot_y = round((padding + 7 + 1) * cell_size)
            mask_corners[top_left_top_y:top_left_bot_y, top_left_left_x:top_left_right_x] = 1

            top_right_left_x = round((padding + qr_size - 8) * cell_size)
            top_right_right_x = round((padding + qr_size + 1) * cell_size)
            top_right_top_y = round((padding - 1) * cell_size)
            top_right_bot_y = round((padding + 7 + 1) * cell_size)
            mask_corners[top_right_top_y:top_right_bot_y, top_right_left_x:top_right_right_x] = 1

            bot_left_left_x = round((padding - 1) * cell_size)
            bot_left_right_x = round((padding + 7 + 1) * cell_size)
            bot_left_top_y = round((padding + qr_size - 8) * cell_size)
            bot_left_bot_y = round((padding + qr_size + 1) * cell_size)
            mask_corners[bot_left_top_y:bot_left_bot_y, bot_left_left_x:bot_left_right_x] = 1

            if qr_size >= 25:  # bottom right anchor only exists in larged qr codes
                bot_right_left_x = round((padding + qr_size - 7 - 2) * cell_size)
                bot_right_right_x = round((padding + qr_size - 7 + 3) * cell_size)
                bot_right_top_y = round((padding + qr_size - 7 - 2) * cell_size)
                bot_right_bot_y = round((padding + qr_size - 7 + 3) * cell_size)
                mask_corners[bot_right_top_y:bot_right_bot_y, bot_right_left_x:bot_right_right_x] = 1

        # mask pixel centers. I'll do full at very center and 0.5 around
        for cell_r in range(qr_size):
            for cell_c in range(qr_size):
                center_x = round((cell_c + padding) * cell_size + cell_size / 2)
                center_y = round((cell_r + padding) * cell_size + cell_size / 2)
                if self.mask[center_y, center_x] == 0:  # haven't masked the corner
                    if bigger:
                        self.mask[center_y - 1:center_y + 2, center_x - 2] = 0.25 * factor
                        self.mask[center_y - 1:center_y + 2, center_x + 2] = 0.25 * factor
                        self.mask[center_y - 2, center_x - 1:center_x + 2] = 0.25 * factor
                        self.mask[center_y + 2, center_x - 1:center_x + 2] = 0.25 * factor
                    self.mask[center_y - 1:center_y + 2, center_x - 1:center_x + 2] = 0.5 * factor
                    self.mask[center_y, center_x] = 1

        self.mask = self.mask[None, ...]  # batch dim
        if self.split:
            self.mask_corners = self.mask_corners[None, ...]

        if False:  # plot
            self.plot(self.get_mask())
            if self.split:
                self.plot(self.get_mask(corner_mask=True))
            pass

        if blur:
            # binary_tensor_plot(self.mask)
            mask = self.mask.detach().numpy() * 255
            mask = ndimage.gaussian_filter(mask, 1.5) / 255 * 1.2
            mask = np.minimum(1, mask)
            self.mask = torch.Tensor(mask)
            # binary_tensor_plot(self.mask)


    @staticmethod
    def plot(mask):
        mask = (mask.squeeze() * 255).astype(np.uint8)
        plt.imshow(mask[:, :, np.newaxis])
        plt.show()

    def get_mask(self, numpy=True, corner_mask=False):
        mask = self.mask_corners if corner_mask else self.mask
        if numpy:
            return mask.detach().numpy().transpose(1, 2, 0)
        else:
            return mask

    def forward(self, pred, gt):
        pred = pred.mean(dim=1)  # grayscale!
        assert (gt.size(1) == 1)
        gt = gt[:, 0]
        diff = torch.abs(pred - gt)
        diff[diff < self.threshold] = 0
        if diff.is_cuda and not self.mask.is_cuda:
            self.mask = self.mask.to(diff.device)
            if self.split:
                self.mask_corners = self.mask_corners.to(diff.device)

        if self.split:
            diff_corners = diff * self.mask_corners
            diff_message = diff * self.mask
            return (diff_corners.sum() / self.mask_corners.sum() + diff_message.sum() / self.mask.sum()) / 2
        else:
            diff *= self.mask
            return diff.mean()
        if False:
            import matplotlib.pyplot as plt
            x = self.mask.detach().cpu().numpy().squeeze()
            plt.imshow((x + 1) * 127.5, cmap="gray");
            plt.show()


import string
import random

MASTER_STRING = string.ascii_letters


def make_QR_code(length=58, size=256, error_level="l"):
    import qrcode
    error_levels = {"l": 1, "m": 0, "q": 3, "h": 2}  # L < M < Q < H

    qr = qrcode.QRCode(
        version=1,
        error_correction=error_levels[error_level],
        box_size=6,
        border=2,
        mask_pattern=1,
    )
    gt_char = ''.join(random.choices(MASTER_STRING, k=length))
    qr.add_data(gt_char)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black",
                        back_color="white").resize([size, size])
    img = np.array(img).astype(np.uint8) * 255
    return img


if __name__ == "__main__":
    size = 256
    length = 34
    error = "h"  # h is best
    qr_img = make_QR_code(length, size, error)
    qr = QRCenterPixelLoss(size, 33, 2, 0.1, bigger=True, split=False, factor=1.5, no_corners=False)
    mask = (qr.get_mask().squeeze() * 255).astype(np.uint8)
    output = (qr_img + mask) / 2
    if False:
        plt.imshow(output[:, :, np.newaxis])
        plt.show()

