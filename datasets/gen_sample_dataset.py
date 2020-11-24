import torch
from torch.utils.data import Dataset
import random

def collate(batch):
    batch = [b for b in batch if b is not None]
    return {'image': torch.stack([b['image'] for b in batch],dim=0),
            'gt_char': [b['gt_char'] for b in batch],
            'targetchar': torch.stack([b['targetchar'] for b in batch], dim=0),
            'targetvalid': torch.cat([b['targetvalid'] for b in batch], dim=0)
            }

class GenSampleDataset(Dataset):
    def __init__(self, batch_size,seed_config,cache_dir):
        self.seed = AdvancedQRDataset2(None,'train',seed_config)
        self.char_to_index = self.seed.char_to_index
        self.char_set_len = len(self.seed.char_to_index)
        self.max_message_len = self.seed.max_message_len

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
        if random.random()<prob_sample:
            img,valid,chars = self.sample_gen(isvalid)
            targetchar = torch.LongTensor(self.char_set_len).fill_(0)
            for i,c in enumerate(chars):
                targetchar[i]=self.char_to_index[c]
            targetvalid = torch.FloatTensor([1])

            return {
                "image": img,
                "gt_char": gt_char,
                'targetchar': targetchar,
                'targetvalid': torch.FloatTensor([valid])
            }
        else:
            return self.seed[random.randrange(len(self.seed))]

    def __len__(self):
        return 256

    def sample_gen(self,isvalid):
        if isvalid[b]:
            saved = self.saved_valid
            saved_cache = self.saved_cache_valid
        else:
            saved = self.saved_invalid
            saved_cache = self.saved_cache_invalid
        images=[]

        targetchar = torch.LongTensor(self.max_message_len).fill_(0)
        i = random.randint(0,len(saved)-1)
        inst = torch.load(saved[i])
        if isvalid:
            image,chars = inst
            for i,c in enumerate(chars):
                targetchar[i]=self.char_to_index[c]
            targetvalid = torch.FloatTensor(batch_size).ones_()
        else:
            image = inst
            targetvalid = torch.FloatTensor(batch_size).zero_()

        
        return image,targetvalid,targetchar

    def add_gen_sample(self,images,isvalid,chars):
        batch_size = images.size(0)
        images = images.cpu().detach()

        for b in range(batch_size):
            if isvalid[b]:
                inst = (images[b],isvalid[b],chars[b])
                saved = self.saved_valid
                saved_cache = self.saved_cache_valid
            else:
                inst = images[b]
                saved = self.saved_invalid
                saved_cache = self.saved_cache_invalid

            if len(saved)>= self.max_stored:
                if random.random() > self.forget_new_freq:
                    change = random.randint(0,len(saved)-1)
                    torch.save(inst,saved[change])
            else:
                path = os.path.join(saved_cache,'{}.pt'.format(len(saved)))
                torch.save(inst,path)
                saved.append(path)
