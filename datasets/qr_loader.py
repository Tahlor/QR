import numpy as np
import qrcode
from torch.utils.data import Dataset
from pathlib import Path

class QRDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_new_qr_code(self, message="", save_location="", **kwargs):
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=11,
            border=2,
            mask_pattern=1
        )

        if not message:
            message = self.create_message()

        qr.add_data(message)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")

        if save_location:
            img.save(Path(save_location) / message)


        return {
                "text": message,
                "encoded_text": message,
                "img": img
               }

    @staticmethod
    def create_message(self, l=15):
        return f"{np.random.randint(0,1e15)}:015d}"

    def __getitem__(self, idx):
        if self.data:
            return self.data[idx]
        else:
            return self.generate_new_qr_code()



TYPE = np.float32 #np.float16
def collate_stroke(batch, device="cpu", ignore_alphabet=False, gt_opts=None, post_length_buffer=20, alphabet_size=0):
    """ Pad ground truths with 0's
        Report lengths to get accurate average loss

        stroke_points_gt : padded with repeated last point
        stroke_points_rel : x_rel, abs_y, SOS, 0's

    Args:
        batch:
        device:

    Returns:

    """
    vocab_size = batch[0]['gt'].shape[-1]
    batch = [b for b in batch if b is not None]
    #These all should be the same size or error
    if len(set([b['line_img'].shape[0] for b in batch])) > 1: # All items should be the same height!
        logger.warning("Problem with collating!!! See hw_dataset.py")
        logger.info(batch)
    assert len(set([b['line_img'].shape[0] for b in batch])) == 1
    assert len(set([b['line_img'].shape[2] for b in batch])) == 1

    batch_size = len(batch)
    dim0 = batch[0]['line_img'].shape[0] # height
    dim1 = max([b['line_img'].shape[1] for b in batch]) # width
    dim2 = batch[0]['line_img'].shape[2] # channel

    max_feature_map_size = max([b['feature_map_width'] for b in batch])
    all_labels_numpy = []
    label_lengths = []
    start_points = []

    # Make input square BATCH, H, W, CHANNELS
    imgs_gt = np.full((batch_size, dim0, dim1, dim2), PADDING_CONSTANT).astype(TYPE)
    max_label = max([b['gt'].shape[0] for b in batch]) # width

    stroke_points_gt = np.full((batch_size, max_label, vocab_size), PADDING_CONSTANT).astype(TYPE)
    stroke_points_rel = np.full((batch_size, max_label+1, vocab_size), 0).astype(TYPE)
    mask = np.full((batch_size, max_label, 1), 0).astype(TYPE)
    feature_map_mask = np.full((batch_size, max_feature_map_size), 0).astype(TYPE)

    # Loop through instances in batch
    for i in range(len(batch)):
        b_img = batch[i]['line_img']
        imgs_gt[i,:,: b_img.shape[1],:] = b_img

        l = batch[i]['gt']
        #all_labels.append(l)
        label_lengths.append(len(l))
        ## ALL LABELS - list of desired_num_of_strokes batch size; arrays LENGTH, VOCAB SIZE
        stroke_points_gt[i, :len(l), :] = l
        stroke_points_gt[i, len(l):, :] = l[-1] # just repeat the last element; this works when using ABS coords for GTs (default) and EOS

        # Relative version - this is 1 indx longer - first one is 0's
        rel_x = stroke_recovery.relativefy_numpy(l[:,0:1])
        stroke_points_rel[i, 1:1+len(l), 0] = rel_x # use relative coords for X, then 0's
        stroke_points_rel[i, 1:1+len(l), 1:2] = stroke_points_gt[i, :len(l), 1:2] # Copy the absolute ones for Y, then 0's
        stroke_points_rel[i, batch[i]['sos_args']+1, 2] = 1 # all 0's => 1's where SOS are
        # No EOS specified for x_rel

        mask[i, :len(l), 0] = 1
        feature_map_mask[i, :batch[i]['feature_map_width']+post_length_buffer] = 1 # keep predicting after

        all_labels_numpy.append(l)
        start_points.append(torch.from_numpy(batch[i]['start_points'].astype(TYPE)).to(device))

    label_lengths = np.asarray(label_lengths)

    line_imgs = imgs_gt.transpose([0,3,1,2]) # batch, channel, h, w
    # print(np.min(line_imgs))
    # plt.hist(line_imgs.flatten())
    # plt.show()

    line_imgs = torch.from_numpy(line_imgs).to(device)
    stroke_points_gt = torch.from_numpy(stroke_points_gt.astype(TYPE)).to(device)
    #label_lengths = torch.from_numpy(label_lengths.astype(np.int32)).to(device)
    stroke_points_gt_rel = torch.from_numpy(stroke_points_rel.astype(TYPE)).to(device)

    mask = torch.from_numpy(mask.astype(TYPE)).to(device)
    feature_map_mask = torch.from_numpy(feature_map_mask.astype(TYPE)).to(device)

    # TEXT STUFF - THIS IS GOOD STUFF
    ## get sequence lengths
    text_lengths = torch.tensor([len(b["gt_text_indices"]) for b in batch])

    ## pad
    if ignore_alphabet:
        one_hot = []
        padded_one_hot = torch.zeros(1)
        text_mask = []
    else:
        one_hot = [torch.nn.functional.one_hot(torch.tensor(t["gt_text_indices"]), alphabet_size) for t in batch]

        # BATCH, MAX LENGTH, ALPHA SIZE
        padded_one_hot = torch.nn.utils.rnn.pad_sequence(one_hot, batch_first=True)

        ## compute mask
        text_mask = (torch.max(padded_one_hot, axis=-1).values != 0)

    return_d = {
        "feature_map_mask": feature_map_mask,
        "mask": mask,
        "gt_text": [b["gt_text"] for b in batch], # encode this
        "gt_text_indices": [b["gt_text_indices"] for b in batch],
        "gt_text_mask": text_mask,
        "gt_text_one_hot": padded_one_hot.to(torch.float32),
        "gt_text_lengths": text_lengths,
        "line_imgs": line_imgs,
        "gt": stroke_points_gt, # Numpy Array, with padding
        "rel_gt": stroke_points_gt_rel,
        "gt_list": [torch.from_numpy(l.astype(TYPE)).to(device) for l in all_labels_numpy], # List of numpy arrays
        #"gt_reverse_strokes": [torch.from_numpy(b["gt_reverse_strokes"].astype(TYPE)).to(device) for b in batch],
        "gt_numpy": all_labels_numpy,
        "start_points": start_points,  # List of numpy arrays
        "gt_format": [batch[0]["gt_format"]]*batch_size,
        "label_lengths": label_lengths,
        "paths":  [b["path"] for b in batch],
        "x_func": [b["x_func"] for b in batch],
        "y_func": [b["y_func"] for b in batch],
        "kdtree": [b["kdtree"] for b in batch],
        "gt_idx": [b["gt_idx"] for b in batch],
        "raw_gt": [b["raw_gt"] for b in batch],
        'id':     [b["id"] for b in batch]
    }

    # Pass everything else through too
    for i in batch[0].keys():
        if i not in return_d.keys():
            return_d[i] = [b[i] for b in batch]

    return return_d
