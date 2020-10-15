# Framework code from handwriting generation project
"Text and Style Conditioned GAN for the Generation of Offline-Handwriting Lines" published at BMVC 2020. https://arxiv.org/abs/2009.00678

## Requirements
* Python 3.x
* PyTorch >1.0
* torchvision
* opencv
* scikit-image
* editdistance


To use deformable convs you'll need to copy this repo in the main directory https://github.com/open-mmlab/mmdetection
Install using this command in the mmdetection directory: `python setup.py develop`
This code only compiles on a GPU version Pytorch, CPU-only does not work.

## Folder Structure
  ```
  
  │
  ├── train.py - Use this to train
  ├── eval.py - Use this to evaluate and save images. Some examples of how to run this in notes.txt. This was used to generate the reconstruction images for the paper.
  ├── graph.py - Display plots given a training snapshot
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py - abstract base class for data loaders (unused?)
  │   ├── base_model.py - abstract base class for models
  │   └── base_trainer.py - abstract base class for trainers
  │
  ├── data/ - this has various files that were convenient to keep with the project
  ├── data_loader/ 
  │   └── data_loaders.py - This just gets you the right dataset
  │
  ├── datasets/ - default datasets folder
  │   ├── hw_dataset.py - basic dataset made to handle IAM data at the line level
  │   └── synth_text_dataset.py - dataset for on-the-fly-but-cached sythetic (font) dataset. This can be used as a basis for a similar QR image dataset
  │
  ├── logger/ - for training process logging
  │   └── logger.py
  │
  ├── model/ - models, losses, and metrics
  │   ├── loss.py - has all losses, here or imported here
  │   ├── qr_wraper.py - holds generator, discrminator, and QRnet models for training (easy saving/loading weights)
  │   ├── MUNIT_networks.py - has code from MUNIT paper as well as my own generator classes
  │   ├── aligned_l1_loss.py - l1 loss after aligning input and target  by sliding them along eachother
  │   ├── attention.py - functions for multi-head dot product attention
  │   ├── cnn_lstm.py - Code from Start, Follow, Read, with minor tweaks to allow Group Norm and logsoftmax at the end
  │   ├── discriminator.py - Has various discriminators I tried, all using spectral norm
  │   ├── elastic_layer.py - Chris's code along with my modification to allow style appending
  │   │
  │   ├── key_loss.py - Defines the loss to cause keys to be alteast some distance from their closest neighbor
  │   ├── net_builder.py - various auxilary network construction functions from previous project. I think I only use "getGroupSize()" to use with Group Norm
  │   ├── pure_gen.py - contains model from paper (PureGen) based on StyleGAN as well as several variations I experimented with
  │   ├── simple_gan.py - Non-StyleGAN based generator, and a simpler discriminator
  │   └── pyramid_l1_loss.py - exactly what it says
  │
  ├── saved/ - default checkpoints folder
  │
  ├── cf_example.json - Example configuration file
  │
  ├── trainer/ - trainers
  │   ├── qr_gen_trainer.py - Has the code to train our generator. Currently assumes pretrained QRnet If we do anything that's not tweakign the model or data, it goes here.
  │   ├── hw_with_style_trainer.py - This has the code to run training for the hw paper
  │   └── trainer.py - basic, should work for image classification stuff
  │
  └── utils/
      ├── util.py - importantly has code to create mask from handwriting image and extact centerline from handwriting image
      ├── augmentation.py - Chris's brightness augmentation
      ├── curriculum.py - this object handles tracking the curriculum during training
      ├── character_set.py - Gets the character set from label files (modfied from Start,Follow,Read code)
      ├── error_rates.py - character error, etc
      ├── grid_distortion.py - Curtis's augmentation
      ├── metainit.py - I tried getting this paper to work: "MetaInit: Initialzing learning by learning to initialize"
      ├── normalize_line.py - functions to noramlize a line image
      ├── string_utils.py - used for converting string characters to their class numbers and back
      └── util.py - various functions
  ```


## Config file format
Config files are in `.json` format. 
  ```
{
    "name": "long_Gskip",                                       #name for saving
    "cuda": true,                                               #use GPUs
    "gpu": 0,                                                   #only single GPU supported
    "save_mode": "state_dict",                                  #can change to save whole model
    "override": true,                                           #if resuming, replace config file
    "super_computer":false,                                     #whether to mute log output
    "data_loader": {
        "data_set_name": "AuthorHWDataset",                     #class name

        "data_dir": "/trainman-mount/trainman-storage-8308c0a4-7f25-47ad-ae22-1de9e3faf4ad",    #IAM loaction on sensei
        "Xdata_dir": "../data/IAM/",
        "batch_size": 1,
        "a_batch_size": 2,
        "shuffle": true,
        "num_workers": 2,

        "img_height": 64,
        "max_width": 1400,
        "char_file": "./data/IAM_char_set.json",
        "mask_post": ["thresh","dilateCircle","errodeCircle"],
        "mask_random": false,
        "spaced_loc": "../saved/spaced/spaced.pkl"
    },
    "validation": {
        "shuffle": false,
        "batch_size": 3,
        "a_batch_size": 2,
        "spaced_loc": "../saved/spaced/val_spaced.pkl"
    },

    
    "lr_scheduler_type": "none",
 
    "optimizer_type": "Adam",
    "optimizer": {
        "lr": 0.00001,
        "weight_decay": 0,
        "betas": [0.5,0.99]
    },
    "optimizer_type_discriminator": "Adam",                 #seperate optimizer for discriminators
    "optimizer_discriminator": {
        "lr": 0.00001,
        "weight_decay": 0,
        "betas": [0,0.9]
    },
    "loss": {                                               #adversarial losses (generative and discriminator) are hard coded and not specified here
        "auto": "pyramidL1Loss",
        "key": "pushMinDist",
        "count": "MSELoss",
        "mask": "HingeLoss",
        "feature": "L1Loss",
        "reconRecog": "CTCLoss",
        "genRecog": "CTCLoss",
        "genAutoStyle": "L1Loss"
    },
    "loss_weights": {                                       #multiplied to loss to balance them
        "auto": 1,
        "discriminator": 0.1,
        "generator": 0.01,
        "key": 0.001,
        "count": 0.1,
        "mask": 0.1,
        "mask_generator": 0.01,
        "mask_discriminator": 0.01,
        "feature": 0.0000001,
        "reconRecog": 0.000001,
        "genRecog": 0.0001,
        "style_discriminator": 0.1,
        "style_generator": 0.01,
        "genAutoStyle": 0.1

    },
    "loss_params":                                          #additional params passed directly to function
        {
            "auto": {"weights":[0.4,0.3,0.3],
                     "pool": "avg"},
            "key": {"dist":"l1",
                    "thresh": 1.0},
            "mask": {"threshold": 4}
        },
    "metrics": [],                                          #unused
    "trainer": {
        "class": "HWWithStyleTrainer",
        "iterations": 700000,                               #Everything is iterations, because epochs are weird.
        "save_dir": "../saved/",
        "val_step": 2000,                                   #how frequently to run through validation set
        "save_step": 5000,                                  #how frequently to save a seperate snapshot of the training & model
        "save_step_minor": 250,                             #how frequently to save a "latest" model (overwrites)
        "log_step": 100,                                    #how frequently to print training stats
        "verbosity": 1,
        "monitor": "loss",
        "monitor_mode": "none",
        "space_input": true,
        "style_together": true,                             #append sty
        "use_hwr_pred_for_style": true,
        "hwr_without_style":true,
        "slow_param_names": ["keys"],
        "curriculum": {
            "0": [["auto"],["auto-disc"]],
            "1000": [["auto", "auto-gen"],["auto-disc"]],
            "80000": [["count","mask","gt_spaced","mask-gen"],["auto-disc","mask-disc"]],
            "100000": [  [1,"count","mask","gt_spaced","mask-gen"],
                        ["auto","auto-gen","count","mask","gt_spaced","mask_gen"],
                        ["auto","auto-mask","auto-gen","count","mask","gt_spaced","mask_gen"],
                        [2,"gen","gen-auto-style"],
                        [2,"disc","mask-disc"],
                        [2,"auto-disc","mask-disc"]]
        },
        "balance_loss": true,
        "interpolate_gen_styles": "extra-0.25",

	"text_data": "data/lotr.txt",

        "use_learning_schedule": false
    },
    "arch": "HWWithStyle", 
    "model": {
        "num_class": 80,
        "generator": "SpacedWithMask",
        "gen_dim": 128,
        "gen_n_res1": 2,
        "gen_n_res2": 3,
        "gen_n_res3": 2,
        "gen_use_skips": true,
	"hwr": "CRNN_group_norm_softmax",
        "pretrained_hwr": "../saved/IAM_hwr_softmax_aug/checkpoint-latest.pth",
        "hwr_frozen": true,
        "style": "new",
        "style_norm":"group",
        "style_activ":"relu",
        "style_dim": 256,
        "num_keys": 64,
        "global_pool": true,
        "discriminator": "two-scale-better more low global",
        "spacer": "duplicates",
        "create_mask": true,
        "mask_discriminator": "with derivitive"

    }
}
  ```

##  Using config files
Modify the configurations in `.json` config files, then run:

  ```
  python train.py --config config.json
  ```

##  Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume path/to/checkpoint
  ```


The checkpoints will be saved in `save_dir/name/`.

The config file is saved in the same folder. (as a reference only, the config is loaded from the checkpoint)

**Note**: checkpoints contain:
  ```python
  {
    'arch': arch,
    'epoch': epoch,
    'logger': self.train_logger,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'monitor_best': self.monitor_best,
    'config': self.config
  }
  ```

