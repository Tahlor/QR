import warnings
import os
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


out = Path("./auto/")
out.mkdir(parents=True, exist_ok=True)
config_prime = edict(json.load(Path("./DEFAULT.json").open()))

if is_galois():
    dataset_root = Path("/media/taylor/Seagate Backup Plus Drive/datasets")
else:
    dataset_root = Path("../data")

datasets = {
            "art": dataset_root / "abstract_art",
            "texture": dataset_root / "texture",
            "cloud": dataset_root / "understanding_cloud_organization",
            "water": dataset_root / "water",
            }

#../data/terrain
# configs/brian/cf_preArt_trainDecoder_newPix_maskCorners.json

for coord_conv in [False]:
    for hi_res in [True, False]:
        for patch_type in "patch", "patch_layers6":
            for dataset in datasets.keys():
                config = deepcopy(config_prime)
                config.loss_params.pixel.no_corners = True

                if not coord_conv:
                    config.model.generator = config.model.generator.replace("coordconv", "")

                config.data_loader.data_dir = str(datasets[dataset])

                if hi_res:
                    config.loss_params.pixel.qr_size = 33
                    config.loss_params.pixel.factor = .1

                    # delete encoder
                    config.model.generator = config.model.generator.replace("predict_offset", "")  # "SG2UGen small coordconv unbound predict_offset"
                    #config.model.discriminator = "StyleGAN2 smaller mask_corners"
                    config.model.pretrained_qr = None  # "saved/default_cconv_digits_multimask/checkpoint-latest.pth"

                    # make hi res QR
                    config.data_loader.QR_dataset.str_len = 34
                    config.data_loader.QR_dataset.data_set_name = "SimpleQRDataset"
                    config.data_loader.alphabet_description = "printable"
                    config.data_loader.distortions = False
                    config.data_loader.alphabet = string.printable
                    config.data_loader.characters = config.data_loader.alphabet
                    config.data_loader.error_level = "h"
                    config.data_loader.min_length = config.data_loader.QR_dataset.str_len - 2
                    # bigger discriminator / generator
                    if not is_galois():
                        config.model.generator = config.model.generator.replace("small","")
                        config.model.discriminator = config.model.discriminator.replace("small", "")

                    config.loss.pop("char")
                    config.loss.pop("valid")
                    for i,item in reversed(list(enumerate(config.trainer.curriculum["0"]))):
                        if item[0] == "decoder":
                            x = config.trainer.curriculum["0"].pop(i)
                            #print(x)

                    config.pop("optimizer_type_decoder")
                    config.pop("optimizer_decoder")
                else:
                    config.loss_params.pixel.factor = .5

                config.loss_params.pixel.threshold = .2
                hi_res = "hi_res" if hi_res else "low_res"
                name = f"{dataset}_{hi_res}_{patch_type}"

                config.trainer.print_dir = str(ROOT / f"train_out/{name}")
                config.sample_data_loader.cache_dir = str(ROOT / f"cache/{name}")

                Path(config.trainer.print_dir).mkdir(parents=True, exist_ok=True)
                Path(config.sample_data_loader.cache_dir).mkdir(parents=True, exist_ok=True)

                config.name = Path(name).stem
                json.dump(config.__dict__,(out / f"___{name}.json").open("w"), indent=4, separators=(',', ':'))
                #os.chmod(out / f"___{name}", 755)

gen_slurm_scripts.run_it(out)
