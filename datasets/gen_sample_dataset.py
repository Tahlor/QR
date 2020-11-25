import torch
from torch.utils.data import Dataset
import random, os
from .advanced_qr_dataset2 import AdvancedQRDataset2
from utils import util

def collate(batch):
    batch = [b for b in batch if b is not None]
    return {'image': torch.stack([b['image'] for b in batch],dim=0),
            'gt_char': [b['gt_char'] for b in batch],
            'targetchar': torch.stack([b['targetchar'] for b in batch], dim=0),
            'targetvalid': torch.cat([b['targetvalid'] for b in batch], dim=0)
            }

class GenSampleDataset(Dataset):
    def __init__(self, config, seed_config):
        self.seed = AdvancedQRDataset2(None,'train',seed_config)
        self.char_to_index = self.seed.char_to_index
        self.char_set_len = len(self.seed.char_to_index)
        self.max_message_len = self.seed.max_message_len

        cache_dir = config['cache_dir']
        self.max_saved= config['max_saved']
        self.still_seed_prob = config['still_seed_prob'] if 'still_seed_prob' in config else 0.005
        self.forget_new_freq = config['forget_new_freq'] if 'forget_new_freq' in config else 0.05

        self.saved_valid=[]
        self.saved_invalid=[]
        self.saved_cache_valid = os.path.join(cache_dir,'valid')
        self.saved_cache_invalid = os.path.join(cache_dir,'invalid')
        util.ensure_dir(self.saved_cache_valid)
        util.ensure_dir(self.saved_cache_invalid)
        for i in range(self.max_saved):
            path = os.path.join(self.saved_cache_valid,'{}.pt'.format(i))
            if os.path.exists(path):
                self.saved_valid.append(path)

            path = os.path.join(self.saved_cache_invalid,'{}.pt'.format(i))
            if os.path.exists(path):
                self.saved_invalid.append(path)

    def __getitem__(self, idx):
        isvalid = random.random()<0.5
        if isvalid:
            prob_sample = (len(self.saved_valid)/self.max_saved) *(1-self.still_seed_prob)
        else:
            prob_sample = (len(self.saved_invalid)/self.max_saved) *(1-self.still_seed_prob)
        if random.random()<prob_sample:
            img,targetvalid,targetchar,chars = self.sample_gen(isvalid)

            return {
                "image": img,
                "gt_char": chars,
                'targetchar': targetchar,
                'targetvalid': targetvalid
            }
        else:
            return self.seed[random.randrange(len(self.seed))]

    def __len__(self):
        return 256

    def sample_gen(self,isvalid):
        if isvalid:
            saved = self.saved_valid
            saved_cache = self.saved_cache_valid
        else:
            saved = self.saved_invalid
            saved_cache = self.saved_cache_invalid
        images=[]

        targetchar = torch.LongTensor(self.max_message_len).fill_(0)
        for i in range(50):
            try: #sometimes write errors occur (model crash?) this smooths these over
                i = random.randint(0,len(saved)-1)
                inst = torch.load(saved[i])
                break
            except:
                continue
        if isvalid:
            image,chars = inst
            for i,c in enumerate(chars):
                targetchar[i]=self.char_to_index[c]
            targetvalid = torch.FloatTensor(1).ones_()
        else:
            image = inst
            targetvalid = torch.FloatTensor(1).zero_()
            chars=None

        
        return image,targetvalid,targetchar,chars

    def add_gen_sample(self,images,isvalid,chars):
        batch_size = images.size(0)
        images = images.cpu().detach()

        for b in range(batch_size):
            if isvalid[b]:
                inst = (images[b],chars[b])
                saved = self.saved_valid
                saved_cache = self.saved_cache_valid
            else:
                inst = images[b]
                saved = self.saved_invalid
                saved_cache = self.saved_cache_invalid

            if len(saved)>= self.max_saved:
                if random.random() > self.forget_new_freq:
                    change = random.randint(0,len(saved)-1)
                    torch.save(inst,saved[change])
            else:
                path = os.path.join(saved_cache,'{}.pt'.format(len(saved)))
                torch.save(inst,path)
                saved.append(path)
