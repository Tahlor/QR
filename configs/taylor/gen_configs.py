import warnings
import os
import shutil
import json
import string
from pathlib import Path
from copy import deepcopy
from easydict import EasyDict as edict
import gen_slurm_scripts
import socket

def get_computer():
    return socket.gethostname()
def is_galois():
    return get_computer() == "Galois"


# configs/brian/cf_preArt_trainDecoder_newPix_maskCorners.json

from time import sleep

ROOT = gen_slurm_scripts.get_root()

VERS="v3"
out = Path(f"./auto_{VERS}/")
try:
    shutil.rmtree(out)
except:
    pass

out.mkdir(parents=True, exist_ok=True)
config_prime = edict(json.load(Path("./DEFAULT2.json").open()))

if is_galois():
    dataset_root = Path("/media/taylor/Seagate Backup Plus Drive/datasets")
else:
    dataset_root = Path("../data")

datasets = {
            "art": dataset_root / "abstract_art",
            "texture": dataset_root / "texture",
            #"cloud": dataset_root / "understanding_cloud_organization",
            "water": dataset_root / "water",
            }

#../data/terrain
# configs/brian/cf_preArt_trainDecoder_newPix_maskCorners.json

for dataset in datasets.keys():
    if dataset in ["art","texture"]:
        patch_list = ["patch_layers6"]
        masked_list = ["maskedInputs", ""]
    else:
        patch_list = ["patch_layer6"]
        masked_list = [""]

    for masked_inputs in masked_list:
        for coord_conv in [False]:
            for hi_res in [True]:
                for patch_type in patch_list:

                    config = deepcopy(config_prime)
                    config.loss_params.pixel.no_corners = False ### MAKE THIS TRUE

                    if not coord_conv:
                        config.model.generator = config.model.generator.replace("coordconv", "")

                    config.data_loader.data_dir = str(datasets[dataset])

                    if hi_res:
                        config.loss_weights.char = 0
                        config.loss_weights.valid = 0
                        config.loss_weights.pixel = 1.2

                        config.loss_params.pixel.qr_size = 33
                        config.loss_params.pixel.factor = 1.5

                        # delete encoder
                        config.model.generator = config.model.generator.replace("predict_offset", "")  # "SG2UGen small coordconv unbound predict_offset"
                        #config.model.discriminator = "StyleGAN2 smaller mask_corners"
                        config.model.pretrained_qr = None  # "saved/default_cconv_digits_multimask/checkpoint-latest.pth"

                        # make hi res QR
                        config.data_loader.QR_dataset.str_len = 34
                        config.data_loader.QR_dataset.data_set_name = "SimpleQRDataset"
                        config.data_loader.QR_dataset.alphabet_description = "printable"
                        config.data_loader.QR_dataset.distortions = False
                        config.data_loader.QR_dataset.alphabet = string.printable
                        config.data_loader.QR_dataset.characters = config.data_loader.QR_dataset.alphabet
                        config.data_loader.QR_dataset.error_level = "h"
                        config.data_loader.QR_dataset.min_message_len = config.data_loader.QR_dataset.str_len
                        config.trainer.modulate_pixel_loss_start = 0
                        config.data_loader.QR_dataset.mask = True if masked_inputs else False
                        config.trainer.retry_count = 10
                        # bigger discriminator / generator
                        if not is_galois():
                            config.model.generator = config.model.generator.replace("small","")
                            config.model.discriminator = config.model.discriminator.replace("small", "")

                        if "char" in config.loss:
                            config.loss.pop("char")
                        if "valid" in config.loss:
                            config.loss.pop("valid")
                        for i,item in reversed(list(enumerate(config.trainer.curriculum["0"]))):
                            if item[0] == "decoder":
                                x = config.trainer.curriculum["0"].pop(i)
                                #print(x)

                        config.pop("optimizer_type_decoder")
                        config.pop("optimizer_decoder")
                    else:
                        config.loss_params.pixel.factor = 1

                    config.loss_params.pixel.threshold = .25
                    hi_res = "hi_res" if hi_res else "low_res"
                    name = f"{VERS}_{dataset}_{hi_res}_{patch_type}_{masked_inputs}"

                    config.trainer.print_dir = str(ROOT / f"train_out/{VERS}/{name}")
                    config.sample_data_loader.cache_dir = str(ROOT / f"cache/{VERS}/{name}")

                    Path(config.trainer.print_dir).mkdir(parents=True, exist_ok=True)
                    Path(config.sample_data_loader.cache_dir).mkdir(parents=True, exist_ok=True)

                    config.name = Path(name).stem
                    json.dump(config.__dict__,(out / f"___{name}.json").open("w"), indent=4, separators=(',', ':'))
                    #os.chmod(out / f"___{name}", 755)

gen_slurm_scripts.run_it(out)
