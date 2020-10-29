from base import BaseModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
#import cv2
#TODO import models and discrimonators
from skimage import draw
from scipy.ndimage.morphology import distance_transform_edt





class QRWraper(BaseModel):
    def __init__(self, config):
        super(QRWraper, self).__init__(config)

        n_downsample = config['style_n_downsample'] if 'style_n_downsample' in config else 3
        input_dim = 1
        dim = config['style_dim']//4 if 'style_dim' in config else 64
        style_dim = config['style_dim'] if 'style_dim' in config else 256
        self.style_dim = style_dim
        norm = config['style_norm'] if 'style_norm' in config else 'none'
        activ = config['style_activ'] if 'style_activ' in config else 'lrelu'
        pad_type = config['pad_type'] if 'pad_type' in config else 'replicate'

        #num_class = config['num_class']
        #self.num_class=num_class

        style_type = config['style'] if 'style' in config else 'normal'

        self.cond_disc=False
        self.vae=False


        qr_type= config['qr_net'] #if 'qr_net' in config else 'default'
        if 'DecoderCNN' in qr_type:
            self.qr_net=DecoderCNN(
        elif 'none' in qr_type:
            self.qr_net=None
        else:
            raise NotImplementedError('unknown QR model: '+qr_type)
        self.qr_frozen=True
        if 'pretrained_qr' in config and config['pretrained_qr'] is not None:
            snapshot = torch.load(config['pretrained_qr'], map_location='cpu')
            qr_state_dict={}
            for key,value in  snapshot['state_dict'].items():
                if key[:4]=='qr_net.':
                    qr_state_dict[key[4:]] = value
            if len(qr_state_dict)==0:
                qr_state_dict=snapshot['state_dict']
            self.qr_net.load_state_dict( qr_state_dict )
            #if 'qr_frozen' in config and config['qr_frozen']:
            #    self.qr_frozen=True
            #    for param in self.qr.parameters():
            #        param.will_use_grad=param.requires_grad
            #        param.requires_grad=False

        if 'generator' in config and config['generator'] == 'none':
            self.generator = None
        elif 'Conv' in config['generator']:
            g_dim = config['gen_dim'] if 'gen_dim' in config else 256
            n_style_trans = config['n_style_trans'] if 'n_style_trans' in config else 6
            down_steps = config['down_steps'] if 'down_steps' in config else 3
            self.generator = ConvGen(num_class,style_dim,g_dim,down_steps,n_style_trans=n_style_trans)
        else:
            raise NotImplementedError('Unknown generator: {}'.format(config['generator']))

        if 'pretrained_generator' in config and config['pretrained_generator'] is not None:
            snapshot = torch.load(config['pretrained_generator'], map_location='cpu')
            gen_state_dict={}
            for key,value in  snapshot['state_dict'].items():
                if key[:10]=='generator.':
                    gen_state_dict[key[10:]] = value
            self.generator.load_state_dict( gen_state_dict )

        if 'discriminator' in config and config['discriminator'] is not None:
            add_noise_img = config['disc_add_noise_img'] if 'disc_add_noise_img' in config else False
            if config['discriminator']=='down':
                self.discriminator = DownDiscriminator()
            elif 'simple' in config['discriminator']:
                self.discriminator=SimpleDisc()
            elif config['discriminator']!='none':
                raise NotImplementedError('Unknown discriminator: {}'.format(config['discriminator']))

        if 'pretrained_discriminator' in config and config['pretrained_discriminator'] is not None:
            snapshot = torch.load(config['pretrained_discriminator'], map_location='cpu')
            discriminator_state_dict={}
            for key,value in  snapshot['state_dict'].items():
                if key[:14]=='discriminator.':
                    discriminator_state_dict[key[14:]] = value
            self.discriminator.load_state_dict( discriminator_state_dict )


    def forward(self,label,label_lengths,style,mask=None,spaced=None,flat=False,center_line=None):
        batch_size = label.size(1)
        if type(self.generator.gen) is NewRNNDecoder:
            if mask is None:
                size = [batch_size,1,64,1200] #TODO not have this hard coded
                dist_map = self.generate_distance_map(size).to(label.device)
            else:
                dist_map = mask

            label_onehot=self.onehot(label)
            if mask is None:
                gen_img = self.generator(label_onehot,style,dist_map,return_intermediate=False)
                self.spacing_pred = None
                self.mask_pred = None
            else:
                gen_img, spaced,mask = self.generator(label_onehot,style,dist_map,return_intermediate=True)
                self.spacing_pred = spaced
                self.mask_pred = mask
        else:
            if mask is None:
                if self.spacer is None:
                    spaced=label
                else:
                    label_onehot=self.onehot(label)
                    self.counts = self.spacer(label_onehot,style)
                    spaced, padded = self.insert_spaces(label,label_lengths,self.counts)
                    spaced = spaced.to(label.device)
                    self.gen_padded = padded
                    if spaced.size(0) > self.max_gen_length:
                        #print('clipping content! {}'.format(spaced.size(0)))
                        diff = self.max_gen_length - spaced.size(0)
                        #cut blanks from the end
                        chars = spaced.argmax(2)
                        for x in range(spaced.size(0)-1,0,-1): #iterate backwards till we meet non-blank
                            if (chars[x]>0).any():
                                break
                        toRemove = min(diff,spaced.size(0)-x+2) #"+2" to pad out a couple blanks
                        if toRemove>0:
                            spaced = spaced[:-toRemove]
                    if spaced.size(0) > self.max_gen_length:
                        diff = self.max_gen_length - spaced.size(0)
                        #cut blanks from the front
                        chars = spaced.argmax(2)
                        for x in range(spaced.size(0)): #iterate forwards till we meet non-blank
                            if (chars[x]>0).any():
                                break
                        toRemove = max(min(diff,x-2),0) #"-2" to pad out a couple blanks
                        if toRemove>0:
                            spaced = spaced[toRemove:]

                        

                    if self.char_style_dim>0:
                        style = self.space_style(spaced,style,spaced.device)
                        self.spaced_style=style

                    if self.cond_disc:
                        self.gen_spaced=spaced
                    if self.create_mask is not None:
                        self.top_and_bottom = self.create_mask(spaced,style)
                        size = [batch_size,1,self.image_height,self.top_and_bottom.size(0)]
                        mask = self.write_mask(self.top_and_bottom,size,flat=flat).to(label.device)
                        if self.clip_gen_mask is not None:
                            mask = mask[:,:,:,:self.clip_gen_mask]
                        self.gen_mask = mask
                    #print('debug. label:{}, spaced:{}, mask:{}'.format(label.size(),spaced.size(),mask.size()))
            elif self.char_style_dim>0:
                if  self.spaced_style is None:
                    style = self.space_style(spaced,style)
                    self.spaced_style=style
                else:
                    style = self.spaced_style #this occurs on an auto-encode with generated mask

            gen_img = self.generator(spaced,style,mask)
        return gen_img

    def autoencode(self,image,label,mask,a_batch_size=None,center_line=None,stop_grad_extractor=False):
        style = self.extract_style(image,label,a_batch_size)
        if stop_grad_extractor:
            if self.char_style_dim>0:
                style = (style[0].detach(),style[1].detach(),style[2].detach())
            else:
                style=style.detach() #This is used when we use the auto-style loss, as wer're using the extractor result as the target
        if self.spaced_label is None:
            self.spaced_label = correct_pred(self.pred,label)
            self.spaced_label = self.onehot(self.spaced_label)
        if type(self.generator.gen) is NewRNNDecoder:
            mask = self.generate_distance_map(image.size(),center_line).to(label.device)
        if mask is None and self.create_mask is not None:
            with torch.no_grad():
                if self.char_style_dim>0:
                    self.spaced_style = self.space_style(self.spaced_label,style)
                    use_style = self.spaced_style
                else:
                    use_style = style
                top_and_bottom =  self.create_mask(self.spaced_label,use_style)
                mask = self.write_mask(top_and_bottom,image.size(),center_line=center_line).to(label.device)
                self.mask = mask.cpu()
        if mask is None and self.create_mask is None:
            mask=0
            ##DEBUG
            #mask_draw = ((mask+1)*127.5).numpy().astype(np.uint8)
            #for b in range(image.size(0)):
            #    cv2.imshow('pred',mask_draw[b,0])
            #    print('mask show')
            #    cv2.waitKey()
        else:
            self.spaced_style = None
        recon = self.forward(label,None,style,mask,self.spaced_label)

        return recon,style

    def extract_style(self,image,label,a_batch_size=None):
        if self.pred is None:
            self.pred = self.hwr(image, None)
        if self.use_hwr_pred_for_style:
            spaced = self.pred.permute(1,2,0)
        else:
            if self.spaced_label is None:
                self.spaced_label = correct_pred(self.pred,label)
                self.spaced_label = self.onehot(self.spaced_label)
            spaced= self.spaced_label.permute(1,2,0)
        batch_size,feats,h,w = image.size()
        if a_batch_size is None:
            a_batch_size = batch_size
        spaced_len = spaced.size(2)
        #append all the instances in the batch by the same author together along the width dimension
        collapsed_image =  image.permute(1,2,0,3).contiguous().view(feats,h,batch_size//a_batch_size,w*a_batch_size).permute(2,0,1,3)
        collapsed_label = spaced.permute(1,0,2).contiguous().view(self.num_class,batch_size//a_batch_size,spaced_len*a_batch_size).permute(1,0,2)
        style = self.style_extractor(collapsed_image, collapsed_label)
        if self.vae:
            if self.char_style_dim>0:
                g_mu,g_log_sigma,spacing_mu, spacing_log_sigma,char_mu,char_log_sigma = style
                g_sigma = torch.exp(g_log_sigma)
                g_style = g_mu + g_sigma * torch.randn_like(g_mu)
                spacing_sigma = torch.exp(spacing_log_sigma)
                spacing_style = spacing_mu + spacing_sigma * torch.randn_like(spacing_mu)
                char_sigma = torch.exp(char_log_sigma)
                char_style = char_mu + char_sigma * torch.randn_like(char_mu)
                self.mu=torch.cat( (g_mu,spacing_mu,char_mu.contiguous().view(batch_size//a_batch_size,-1)), dim=1)
                self.sigma=torch.cat( (g_sigma,spacing_sigma,char_sigma.view(batch_size//a_batch_size,-1)), dim=1)
                style = (g_style,spacing_style,char_style)
            else:
                mu,log_sigma = style
                #assert(not torch.isnan(mu).any())
                assert(not torch.isnan(log_sigma).any())
                if self.training:
                    sigma = torch.exp(log_sigma)
                    style = mu + sigma * torch.randn_like(mu)
                    self.mu=mu
                    self.sigma=sigma
                else:
                    sigma = torch.exp(log_sigma)
                    style = mu + sigma * torch.randn_like(mu)*0.8
                    self.mu=mu
                    self.sigma=sigma
        if self.noisy_style:
            if self.char_style_dim>0:
                raise NotImplementedError('haven;t implmented noise for char spec style vectors')
            else:
                var = style.abs().mean()
                style = style+torch.randn_like(style)*var
        if self.char_style_dim>0:
            g_style,spacing_style,char_style = style
            g_style = torch.cat([g_style[i:i+1].repeat(a_batch_size,1) for i in range(g_style.size(0))],dim=0)
            spacing_style = torch.cat([spacing_style[i:i+1].repeat(a_batch_size,1) for i in range(spacing_style.size(0))],dim=0)
            char_style = torch.cat([char_style[i:i+1].repeat(a_batch_size,1,1) for i in range(char_style.size(0))],dim=0)
            style = (g_style,spacing_style,char_style)
        else:
            #style = style.repeat(a_batch_size,1)
            style = torch.cat([style[i:i+1].repeat(a_batch_size,1) for i in range(style.size(0))],dim=0)
        return style

    def insert_spaces(self,label,label_lengths,counts):
        max_count = max(math.ceil(counts.max()),3)
        lines = []
        max_line_len=0
        batch_size = label.size(1)
        for b in range(batch_size):
            line=[]
            for i in range(label_lengths[b]):
                count = round(np.random.normal(counts[i,b,0].item(),self.count_std))
                if self.count_duplicates:
                    duplicates = round(np.random.normal(counts[i,b,1].item(),self.dup_std))
                else:
                    duplicates=1
                line+=[0]*count + [label[i][b]]*duplicates
            max_line_len = max(max_line_len,len(line))
            lines.append(line)

        spaced = torch.zeros(max_line_len+max_count,batch_size,self.num_class)
        padded=[]
        for b in range(batch_size):
            for i,cls in enumerate(lines[b]):
                spaced[i,b,cls]=1
            for i in range(len(lines[b]),spaced.size(0)):
                spaced[i,b,0]=1
            padded.append((spaced.size(0)-len(lines[b]))/spaced.size(0))

        return spaced, padded

    def write_mask(self,top_and_bottom,size, center_line=None,flat=False):
        #generate a center-line
        batch_size, ch, height, width = size
        mask = torch.zeros(*size)

        if center_line is None:
            center = height//2
            max_center = center+int(height*0.2)
            min_center = center-int(height*0.2)
            step = 3*height/2 #this is from utils.util.getCenterValue()
            last_x = 0
            if flat:
                last_y = np.full(batch_size, center)
            else:
                last_y = np.random.normal(center, (center-min_center)/3, batch_size)
            last_y[last_y>max_center]=max_center
            last_y[last_y<min_center]=min_center
            while last_x<width:
                if flat:
                    next_x=last_x+step
                    next_y=np.full(batch_size, center)
                else:
                    next_x = np.random.normal(last_x+step,step*0.2)
                    next_y = np.random.normal(last_y,(center-min_center)/5,batch_size)
                next_y[next_y>max_center]=max_center
                next_y[next_y<min_center]=min_center

                self.draw_section(last_x,last_y,next_x,next_y,mask,top_and_bottom)

                last_x=next_x
                last_y=next_y
        else:
            ###DEBUG
            if center_line.size(1)<width:
                center_line = torch.cat((center_line, torch.FloatTensor(batch_size,width-center_line.size(1)).fill_(height//2)),dim=1)
            for x in range(width):
                if x>=width or x>=top_and_bottom.size(0):
                    break
                for b in range(batch_size):
                    top = max(0,int(center_line[b,x]-top_and_bottom[x,b,0].item()))
                    bot = min(height,int(center_line[b,x]+top_and_bottom[x,b,1].item()+1))
                    mask[b,0,top:bot,x]=1
        
        blur_kernel = 31
        blur_padding = blur_kernel // 2
        blur = torch.nn.AvgPool2d((blur_kernel//4,blur_kernel//4), stride=1, padding=(blur_padding//4,blur_padding//4))
        return blur((2*mask)-1)

    def draw_section(self,last_x,last_y,next_x,next_y,mask,top_and_bottom):
        batch_size, ch, height, width = mask.size()
        for x in range(int(last_x),int(next_x)):
            if x>=width or x>=top_and_bottom.size(0):
                break
            progress = (x-int(last_x))/(int(next_x)-int(last_x))
            y = (1-progress)*last_y + progress*next_y

            for b in range(batch_size):
                top = max(0,int(y[b]-top_and_bottom[x,b,0].item()))
                bot = min(height,int(y[b]+top_and_bottom[x,b,1].item()+1))
                mask[b,0,top:bot,x]=1



    #def onehot(self,label):
    #    label_onehot = torch.zeros(label.size(0),label.size(1),self.num_class)
    #    #label_onehot[label]=1
    #    for i in range(label.size(0)):
    #        for j in range(label.size(1)):
    #            label_onehot[i,j,label[i,j]]=1
    #    return label_onehot.to(label.device)
    def onehot(self,label): #tensorized version
        label_onehot = torch.zeros(label.size(0),label.size(1),self.num_class)
        label_onehot_v = label_onehot.view(label.size(0)*label.size(1),self.num_class)
        label_onehot_v[torch.arange(0,label.size(0)*label.size(1)),label.view(-1).long()]=1
        return label_onehot.to(label.device)


    def generate_distance_map(self,size,center_line=None):
        batch_size = size[0]
        height = size[2]
        width = size[3]
        line_im = np.ones((batch_size,height,width))

        if center_line is None:
            center = height//2
            max_center = center+int(height*0.2)
            min_center = center-int(height*0.2)
            step = 3*height/2 #this is from utils.util.getCenterValue()
            last_x = 0
            last_y = np.random.normal(center, (center-min_center)/3, batch_size)
            last_y[last_y>max_center]=max_center
            last_y[last_y<min_center]=min_center
            #debug=''
            while last_x<width-1:
                next_x = min(np.random.normal(last_x+step,step*0.2), width-1)
                next_y = np.random.normal(last_y,(center-min_center)/5,batch_size)
                next_y[next_y>max_center]=max_center
                next_y[next_y<min_center]=min_center
                #debug+='{}, '.format(next_y)

                #self.draw_section(last_x,last_y,next_x,next_y,mask,top_and_bottom)
                for b in range(batch_size):
                    rr,cc = draw.line(int(last_y[b]),int(last_x),int(next_y[b]),int(next_x))
                    line_im[b,rr,cc]=0

                last_x=next_x
                last_y=next_y
            #print(debug)
        else:
            for x in range(width):
                if x>=center_line.size(1):
                    break
                #TODO there should be a way to tensorize this
                for b in range(batch_size):
                    line_im[b,round(center_line[b,x].item()),x]=0
        maps=[]
        for b in range(batch_size):
            maps.append(torch.from_numpy(distance_transform_edt(line_im[b])).float())
        maps = torch.stack(maps,dim=0)[:,None,:,:]

        maps /= height/2
        maps[maps>1] = 1
        masp = 1-maps

        return maps

    def space_style(self,spaced,style,device=None):
        #spaced is Width x Batch x Channel
        g_style,spacing_style,char_style = style
        if self.emb_char_style is None:
            device = spaced.device
        spacing_style = spacing_style.to(device)
        char_style = char_style.to(device)
        batch_size = spaced.size(1)
        style = torch.FloatTensor(spaced.size(0),batch_size,self.char_style_dim).to(device)
        text_chars = spaced.argmax(dim=2)
        spacing_style = spacing_style[None,:,:] #add temporal dim for broadcast
        #Put character styles in appropriate places. Fill in rest with projected global style
        for b in range(batch_size):
            lastChar = -1
            for x in range(0,text_chars.size(0)):
                if text_chars[x,b]!=0:
                    charIdx = text_chars[x,b]
                    if self.emb_char_style is not None:
                        style[x,b,:] = self.emb_char_style[charIdx](char_style[b,charIdx])
                        style[lastChar+1:x,b,:] = self.emb_char_style[0](spacing_style[:,b]) #broadcast
                    else:
                        style[x,b,:] = char_style[b,charIdx]
                        style[lastChar+1:x,b,:] = spacing_style[:,b] #broadcast
                    lastChar=x
            style[lastChar+1:,b,:] = spacing_style[:,b]
        return (g_style,style,char_style)
