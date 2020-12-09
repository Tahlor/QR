import os
import json
import string
from pathlib import Path
from copy import deepcopy
from easydict import EasyDict as edict
import gen_scripts

# configs/brian/cf_preArt_trainDecoder_newPix_maskCorners.json

from time import sleep

out = Path("./auto/")
out.mkdir(parents=True, exist_ok=True)
config_prime = edict(json.load(Path("./DEFAULT.json").open()))

datasets = {"art":"../data/abstract_art",
            "texture":"../data/texture",
            "cloud":"../data/understanding_cloud_organization",
            "water":"../data/water",
            }

#../data/terrain
# configs/brian/cf_preArt_trainDecoder_newPix_maskCorners.json

for coord_conv in False:
    for hi_res in True, False:
        for patch_type in "patch", "patch layers6":
            for dataset in datasets.keys():
                config = deepcopy(config_prime)
                config.data_loader.data_dir = datasets[dataset]

                if hi_res:
                    # delete encoder
                    config.model.generator = config.model.generator.replace("predict_offset")  # "SG2UGen small coordconv unbound predict_offset"
                    #config.model.discriminator = "StyleGAN2 smaller mask_corners"
                    config.model.pretrained_qr = None  # "saved/default_cconv_digits_multimask/checkpoint-latest.pth"

                    # make hi res QR
                    config.data_loader.QR_dataset.str_len = 58
                    config.data_loader.data_set_name = "AdvancedQRDataset"
                    config.data_loader.alphabet_description = "printable"
                    config.data_loader.distortions = False
                    config.data_loader.alphabet = string.printable
                    config.data_loader.error_level = "h"

                    # bigger discriminator / generator
                    config.data_loader.generator = config.data_loader.generator.replace("small","")
                    config.data_loader.discriminator = config.data_loader.discriminator.replace("small", "")

                    del config.loss.char
                    del config.loss.valid
                    for i,item in enumerate(config.trainer.curriculum["0"][::-1]):
                        if i == "decoder":
                            config.trainer.curriculum["0"][::-1].pop(i)

                hi_res = "hi_res" if hi_res else "low_res"
                name = f"{dataset}_{hi_res}"

                trainer.print_dir = f"train_out/{name}"
                sample_data_loader.cache_dir = f"cache/{name}"
                Path(trainer.print_dir).mkdir(parents=True, exist_ok=True)
                Path(sample_data_loader.cache_dir).mkdir(parents=True, exist_ok=True)

                config.name = Path(name).stem
                json.dump(config.__dict__,(out / f"___{name}").open("w"), indent=4, separators=(',', ':'))
                #os.chmod(out / f"___{name}", 755)

gen_slurm_scripts.run_it()
