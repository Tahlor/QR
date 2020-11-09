import os
import json
import string
from pathlib import Path
from copy import deepcopy
from easydict import EasyDict as edict
import gen_scripts

out = Path("./auto/")
out.mkdir(parents=True, exist_ok=True)
config_prime = edict(json.load(Path("./DEFAULT.conf").open()))

for coord_conv in True,False:
    for alphabet in [string.digits,string.digits+string.ascii_lowercase]:
        for architecture in ["resnet", "default"]:
            config = deepcopy(config_prime)
            if architecture == "default":
                config.arch = "DecoderCNN"
                config.model = {
                    "input_size": [128, 128],
                    "cnn_layer_specs": [1, 32, "cc-32", "M", "cc-64", "cc-64", "M", "cc-128", "cc-128", "cc-128", "M",
                                        "cc-256", "cc-256", "cc-256"],
                    "fully_connected_specs": ["FC2048", "FC1024"],
                    "max_message_len": 17,
                    "num_char_class": 11
                }
                config.data_loader.coordconv = False
                if not coord_conv:
                    config.model.cnn_layer_specs = [1,32,32,"M",64,64,"M",128,128,128,"M",256,256,256]
            else:
                config.data_loader.coordconv = coord_conv
                config.arch = "ResNet"
            config.data_loader.alphabet = alphabet
            config.model.max_message_len = 17
            config.data_loader.max_message_length = config.model.max_message_len = 17
            config.data_loader.input_size = config.model.input_size
            config.data_loader.batch_size = 32
            name = f"{architecture}{'_cconv' if coord_conv else '_'}_{'alphanumeric' if 'a' in alphabet else 'digits'}.json"
            config.name = Path(name).stem
            json.dump(config.__dict__,(out / f"___{name}").open("w"), indent=4, separators=(',', ':'))
            os.chmod(out / f"___{name}", 755)

gen_scripts.run_it()