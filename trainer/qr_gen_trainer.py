import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from base import BaseTrainer
import timeit
from utils import util, string_utils, error_rates
from utils.metainit import metainitRecog
from data_loader import getDataLoader
from collections import defaultdict
import random, json, os
from model.clear_grad import ClearGrad
#from model.autoencoder import Encoder, EncoderSm, Encoder2, Encoder3, Encoder32
import torchvision.utils as vutils
import utils.img_f as img_f

from model.style_gan2_losses import d_logistic_loss, d_r1_loss, g_nonsaturating_loss, g_path_regularize

def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]

class QRGenTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """
    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(QRGenTrainer, self).__init__(model, loss, metrics, resume, config, train_logger)
        assert(self.curriculum)
        self.config = config
        if data_loader is not None:
            batch_size_size = data_loader.batch_size
            self.data_loader = data_loader
            if 'refresh_data' in dir(self.data_loader.dataset):
                self.data_loader.dataset.refresh_data(None,None,self.logged)
            self.data_loader_iter = iter(data_loader)
        if self.val_step<0:
            self.valid_data_loader=None
            print('Set valid_data_loader to None')
        else:
            self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False



        self.feature_loss = 'feature' in self.loss
        if 'feature' in self.loss:
            self.model.something.setup_save_features()

        self.to_display={}



        self.gan_loss = 'discriminator' in config['model']
        self.disc_iters = config['trainer']['disc_iters'] if 'disc_iters' in config['trainer'] else 1

        #This text data could be used to randomly sample strings, if we so choose
        text_data_batch_size = config['trainer']['text_data_batch_size'] if 'text_data_batch_size' in config['trainer'] else self.config['data_loader']['batch_size']
        text_words = config['trainer']['text_words'] if 'text_words' in config['trainer'] else False
        if 'a_batch_size' in self.config['data_loader']:
            self.a_batch_size = self.config['data_loader']['a_batch_size']
            text_data_batch_size*=self.config['data_loader']['a_batch_size']
        else:
            self.a_batch_size=1
        #text_data_max_len = config['trainer']['text_data_max_len'] if 'text_data_max_len' in config['trainer'] else 20
        if data_loader is not None:
            if 'text_data' in config['trainer']:
                text_data_max_len = self.data_loader.dataset.max_len()
                characterBalance = config['trainer']['character_balance'] if 'character_balance' in config['trainer'] else False
                text_data_max_len = config['trainer']['text_data_max_len'] if 'text_data_max_len' in config['trainer'] else text_data_max_len
                self.text_data = TextData(config['trainer']['text_data'],config['data_loader']['char_file'],text_data_batch_size,max_len=text_data_max_len,words=text_words,characterBalance=characterBalance) if 'text_data' in config['trainer'] else None

        self.balance_loss = config['trainer']['balance_loss'] if 'balance_loss' in config['trainer'] else False # balance the CTC loss with others as in https://arxiv.org/pdf/1903.00277.pdf, although many of may variations (which are better)
        if self.balance_loss:
            self.parameters = list(model.generator.parameters()) #we only balance the generator weights
            self.balance_var_x = config['trainer']['balance_var_x'] if 'balance_var_x' in config['trainer'] else None
            if self.balance_loss.startswith('sign_preserve_x'):
                self.balance_x = float(self.balance_loss[self.balance_loss.find('x')+1:])
            self.saved_grads = [] #this will hold the gradients for previous training steps if "no-step" is specified





        if 'align_network' in config['trainer']:
            self.align_network = JoinNet()
            weights = config['trainer']['align_network']
            state_dict=torch.load(config['trainer']['align_network'], map_location=lambda storage, location: storage)
            self.align_network.load_state_dict(state_dict)
            self.align_network.set_requires_grad(False)

        self.no_bg_loss= config['trainer']['no_bg_loss'] if 'no_bg_loss' in config else False
        
        self.sample_disc = self.curriculum.sample_disc if self.curriculum is not None else False
        #if we are going to sample images from the past for the discriminator, these are to store previous generations
        if self.sample_disc:
            self.new_gen=[]
            self.old_gen=[]
            self.store_new_gen_limit = 10
            self.store_old_gen_limit = config['trainer']['store_old_gen_limit'] if 'store_old_gen_limit' in config['trainer'] else 200
            self.new_gen_freq = config['trainer']['new_gen_freq'] if 'new_gen_freq' in config['trainer'] else 0.7
            self.forget_new_freq = config['trainer']['forget_new_freq'] if 'forget_new_freq'  in config['trainer'] else 0.0
            self.old_gen_cache = config['trainer']['old_gen_cache'] if 'old_gen_cache' in config['trainer'] else os.path.join(self.checkpoint_dir,'old_gen_cache')
            if self.old_gen_cache is not None:
                util.ensure_dir(self.old_gen_cache)
                #check for files in cache, so we can resume with them
                for i in range(self.store_old_gen_limit):
                    path = os.path.join(self.old_gen_cache,'{}.pt'.format(i))
                    if os.path.exists(path):
                        self.old_gen.append(path)
                    else:
                        break


        self.WGAN = config['trainer']['WGAN'] if 'WGAN' in config['trainer'] else False
        self.DCGAN = config['trainer']['DCGAN'] if 'DCGAN' in config['trainer'] else False
        if self.DCGAN:
            self.criterion = torch.nn.BCELoss()

        #if 'encoder_weights' in config['trainer']:
        #    snapshot = torch.load(config['trainer']['encoder_weights'],map_location='cpu')
        #    encoder_state_dict={}
        #    for key,value in  snapshot['state_dict'].items():
        #        if key[:8]=='encoder.':
        #            encoder_state_dict[key[8:]] = value
        #    if 'encoder_type' not in config['trainer'] or config['trainer']['encoder_type']=='normal':
        #        self.encoder = Encoder()
        #    elif config['trainer']['encoder_type']=='small':
        #        self.encoder = EncoderSm()
        #    elif config['trainer']['encoder_type']=='2':
        #        self.encoder = Encoder2()
        #    elif config['trainer']['encoder_type']=='2tight':
        #        self.encoder = Encoder2(32)
        #    elif config['trainer']['encoder_type']=='2tighter':
        #        self.encoder = Encoder2(16)
        #    elif config['trainer']['encoder_type']=='3':
        #        self.encoder = Encoder3()
        #    elif config['trainer']['encoder_type']=='32':
        #        self.encoder = Encoder32(256)
        #    else:
        #        raise NotImplementedError('Unknown encoder type: {}'.format(config['trainer']['encoder_type']))
        #    self.encoder.load_state_dict( encoder_state_dict )
        #    if self.with_cuda:
        #        self.encoder = self.encoder.to(self.gpu)

        #This is for saving results during training!
        self.print_dir = config['trainer']['print_dir'] if 'print_dir' in config['trainer'] else None
        if self.print_dir is not None:
            util.ensure_dir(self.print_dir)
        self.print_every = config['trainer']['print_every'] if 'print_every' in config['trainer'] else 100
        self.iter_to_print = self.print_every
        self.serperate_print_every = config['trainer']['serperate_print_every'] if 'serperate_print_every' in config['trainer'] else 2500
        self.last_print_images=defaultdict(lambda: 0)
        self.print_next_gen=False
        self.print_next_auto=False



        #StyleGAN2 parameters
        self.StyleGAN2 = True
        self.path_batch_shrink = 2
        self.mixing = 0.9
        self.channel_multiplier=2
        self.r1=10
        self.path_regularize=2
        self.mean_path_length=0


        self.modulate_pixel_loss = config['trainer']['modulate_pixel_loss'] if 'modulate_pixel_loss' in config['trainer'] else None
        self.modulate_pixel_loss_start=1000 if 'modulate_pixel_loss_start' not in config['trainer'] else config['trainer']['modulate_pixel_loss_start']
        if self.modulate_pixel_loss=='momentum':
            self.pixel_momentum_B_good=0.9 if 'pixel_momentum_B' not in config['trainer'] else config['trainer']['pixel_momentum_B']
            self.pixel_momentum_B_bad = self.pixel_momentum_B_good*0.4
            self.proper_accept=0.99 if 'proper_accept' not in config['trainer'] else config['trainer']['proper_accept']
            self.pixel_weight_delta=0
            self.pixel_weight_rate=0.1
            self.pixel_thresh_delta=0
            self.pixel_thresh_rate=0.01

            self.max_pixel_weight = 2.5 if 'max_pixel_weight' not in config['trainer'] else float(config['trainer']['max_pixel_weight'])
            self.min_pixel_weight = 0.1 if 'min_pixel_weight' not in config['trainer'] else float(config['trainer']['min_pixel_weight'])
        elif self.modulate_pixel_loss=='bang':
            self.proper_accept=0.99 if 'proper_accept' not in config['trainer'] else config['trainer']['proper_accept']

        self.i_cant=False

        self.combine_qr_loss = False if 'combine_qr_loss' not in config['trainer'] else config['trainer']['combine_qr_loss']


    def _to_tensor(self, data):
        if self.with_cuda:
            #image = data['image'].to(self.gpu)
            qr_image = data['qr_image'].to(self.gpu)
            targetvalid = data['targetvalid'].to(self.gpu)
            targetchar = data['targetchar'].to(self.gpu)
        else:
            #image = data['image']
            qr_image = data['qr_image']
            targetvalid = data['targetvalid']
            targetchar = data['targetchar']
        return qr_image, targetvalid,targetchar


    #I don't use this for metrics, I find it easier to just compute them in run()
    def _eval_metrics(self, typ,name,output, target):
        if len(self.metrics[typ])>0:
            #acc_metrics = np.zeros(len(self.metrics[typ]))
            met={}
            cpu_output=[]
            for pred in output:
                cpu_output.append(output.cpu().data.numpy())
            target = target.cpu().data.numpy()
            for i, metric in enumerate(self.metrics[typ]):
                met[name+metric.__name__] = metric(cpu_output, target)
            return acc_metrics
        else:
            #return np.zeros(0)
            return {}


    def _train_iteration(self, iteration):
        """
        Training logic for an iteration

        :param iteration: Current training iteration.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        #self.model.eval()
        #print("WARNING EVAL")

        #t#tic=timeit.default_timer()#t#
        lesson =  self.curriculum.getLesson(iteration)
        if 'alt-data' in lesson:
            try:
                instance = self.alt_data_loader_iter.next()
            except StopIteration:
                if 'refresh_data' in dir(self.alt_data_loader.dataset):
                    self.alt_data_loader.dataset.refresh_data(None,None,self.logged)
                self.alt_data_loader_iter = iter(self.alt_data_loader)
                instance = self.alt_data_loader_iter.next()
        elif self.curriculum and 'triplet-style' in lesson:
            try:
                instance = self.triplet_data_loader_iter.next()
            except StopIteration:
                if 'refresh_data' in dir(self.triplet_data_loader.dataset):
                    self.triplet_data_loader.dataset.refresh_data(None,None,self.logged)
                self.triplet_data_loader_iter = iter(self.triplet_data_loader)
                instance = self.triplet_data_loader_iter.next()
        else:
            try:
                instance = self.data_loader_iter.next()
            except StopIteration:
                if 'refresh_data' in dir(self.data_loader.dataset):
                    self.data_loader.dataset.refresh_data(None,None,self.logged)
                self.data_loader_iter = iter(self.data_loader)
                instance = self.data_loader_iter.next()
        ##toc=timeit.default_timer()
        ##print('data: '+str(toc-tic))

        ##tic=timeit.default_timer()

        self.optimizer.zero_grad()
        if any(['disc' in l or 'author-train' in l for l in lesson]):
            self.optimizer_discriminator.zero_grad()

        ##toc=timeit.default_timer()
        ##print('for: '+str(toc-tic))

        ##tic=timeit.default_timer()
        #if all([l==0 for l in instance['label_lengths']]):
        #        return {}

        #if (self.iter_to_print<=0 or self.print_next_gen) and 'gen' in lesson:
        #    losses,got = self.run(instance,lesson,get=['gen','disc'])
        #    self.print_images(got['gen'],instance['gt'],got['disc'],typ='gen')
        #    if self.iter_to_print>0:
        #        self.print_next_gen=False
        #    else:
        #        self.print_next_auto=True
        #        self.iter_to_print=self.print_every
        #elif (self.iter_to_print<=0 or self.print_next_auto) and 'auto' in lesson:
        #    losses,got = self.run(instance,lesson,get=['recon','recon_gt_mask'])
        #    self.print_images(got['recon'],instance['gt'],typ='recon',gtImages=instance['image'])
        #    self.print_images(got['recon_gt_mask'],instance['gt'],typ='recon_gt_mask')
        #    if self.iter_to_print>0:
        #        self.print_next_auto=False
        #    else:
        #        self.print_next_gen=True
        #        self.iter_to_print=self.print_every

        if self.iter_to_print<=0:
            losses,run_log,got = self.run(instance,lesson,get=['gen'])
            self.print_images(got['gen'],instance['gt_char'],gtImages=instance['qr_image'],typ='gen')
            self.iter_to_print=self.print_every
        else:
            losses,run_log = self.run(instance,lesson)
            self.iter_to_print-=1
        pred=None

        if losses is None:
            return {}
        #TODO rewrite these for the losses we need to balance (probably just QR reading and validation)
        loss=0
        charLoss=0
        validLoss=0
        pixelLoss=0
        #authorClassLoss=0
        #autoGenLoss=0
        for name in losses.keys():
            losses[name] *= self.lossWeights[name[:-4]]
            #if self.balance_loss and 'generator' in name and 'auto-gen' in lesson:
            #    autoGenLoss += losses[name]
            #elif self.balance_loss and 'Recog' in name:
            #    charLoss += losses[name]
            #elif self.balance_loss and 'authorClass' in name:
            #    authorClassLoss += losses[name]
            if self.balance_loss and 'char' in name:
                charLoss += losses[name]
            elif self.balance_loss and 'valid' in name:
                validLoss += losses[name]
            elif self.balance_loss and 'pixel' in name:
                pixelLoss += losses[name]
            else:
                loss += losses[name]
            losses[name] = losses[name].item()
        if self.combine_qr_loss:
            losses_to_balance = [charLoss+validLoss,pixelLoss] #magic order for balance_var_x param
        else:
            losses_to_balance = [charLoss,validLoss,pixelLoss] #magic order for balance_var_x param
        if (loss!=0 and (torch.isnan(loss) or torch.isinf(loss))):
            print(losses)
        assert(loss==0 or (not torch.isnan(loss) and not torch.isinf(loss)))
        #if pred is not None:
        #    pred = pred.detach().cpu().numpy()
        if type(loss) is not int:
            loss_item = loss.item()
        else:
            loss_item = loss

        if self.balance_loss:
            for b_loss in losses_to_balance:
                if type(b_loss) is not int:
                    saved_grad=[]
                    loss_item += b_loss.item()
                    
                    b_loss.backward(retain_graph=True)

                    for p in self.parameters:
                        if p.grad is not None:
                            #if p.grad.is_cuda:
                            #    saved_grad.append(p.grad.cpu())
                            #else:
                            saved_grad.append(p.grad.clone())
                            p.grad.zero_()
                        else:
                            saved_grad.append(None)
                    self.saved_grads.append(saved_grad)

        else:
            for b_loss in losses_to_balance:
                loss += b_loss


        if type(loss) is not int:
            loss.backward()
            #for p in self.model.parameters():
            #    if p.grad is not None:
            #        assert(not torch.isnan(p.grad).any())

        if self.balance_loss and "no-step" in lesson: #no-step is to split computation over multiple iterations. Disc does step
            saved_grad=[]
            for p in self.parameters:
                if p.grad is None:
                    saved_grad.append(None)
                else:
                    #if p.grad.is_cuda:
                    #    saved_grad.append(p.grad.cpu())
                    #else:
                    saved_grad.append(p.grad.clone())
                    p.grad.zero_()
            self.saved_grads.append(saved_grad)

        elif self.balance_loss and len(self.saved_grads)>0:
            if 'sign_preserve' in self.balance_loss:
                abmean_Ds=[]
                nonzero_sum=0.0
                nonzero_count=0
                for p in self.parameters:
                    if p.grad is not None:
                        abmean_D = torch.abs(p.grad).mean()
                        abmean_Ds.append(abmean_D)
                        if abmean_D!=0:
                            nonzero_sum+=abmean_D
                            nonzero_count+=1
                    else:
                        abmean_Ds.append(None)
                #incase on zero mean
                if nonzero_count>0:
                    nonzero=nonzero_sum/nonzero_count
                    for i in range(len(abmean_Ds)):
                        if abmean_Ds[i]==0.0:
                            abmean_Ds[i]=nonzero
            if self.balance_loss.startswith('sign_preserve_var'):
                sum_multipliers=1
                for iterT,mult in self.balance_var_x.items():
                    if int(iterT)<=iteration:
                        multipliers=mult
                        if type(multipliers) is not list:
                            multipliers=[multipliers]
                multipliers = [self.lossWeights[x] if type(x) is str else x for x in multipliers]
                sum_multipliers+=sum(multipliers)
            for gi,saved_grad in enumerate(self.saved_grads):
                if self.balance_loss.startswith('sign_preserve_var'):
                    x=multipliers[gi]
                for i,(R, p) in enumerate(zip(saved_grad, self.parameters)):
                    if R is not None:
                        #R=R.to(p.device)
                        assert(not torch.isnan(p.grad).any())
                        if self.balance_loss=='sign_preserve': #This is good, although it assigns everything a weight of 1
                            abmean_R = torch.abs(p.grad).mean()
                            if abmean_R!=0:
                                p.grad += R*(abmean_Ds[i]/abmean_R)
                        elif self.balance_loss=='sign_match':
                            match_pos = (p.grad>0)*(R>0)
                            match_neg = (p.grad<0)*(R<0)
                            not_match = ~(match_pos+match_neg)
                            p.grad[not_match] = 0 #zero out where signs don't match
                        elif self.balance_loss=='sign_preserve_fixed':
                            abmean_R = torch.abs(R).mean()
                            if abmean_R!=0:
                                p.grad += R*(abmean_Ds[i]/abmean_R)
                        elif self.balance_loss.startswith('sign_preserve_var'): #This is the best, as you can specify a weight for each balanced term
                            abmean_R = torch.abs(R).mean()
                            if abmean_R!=0:
                                p.grad += x*R*(abmean_Ds[i]/abmean_R)
                        elif self.balance_loss.startswith('sign_preserve_x'):
                            abmean_R = torch.abs(R).mean()
                            if abmean_R!=0:
                                p.grad += self.balance_x*R*(abmean_Ds[i]/(abmean_R+1e-40))
                        elif self.balance_loss=='orig':
                            if R.nelement()>16:
                                mean_D = p.grad.mean()
                                mean_R = R.mean()
                                std_D = p.grad.std()
                                std_R = R.std()
                                if std_D==0 and std_R==0:
                                    ratio=1
                                else:
                                    if std_R==0:
                                        std_R = 0.0000001
                                    ratio = std_D/std_R
                                if ratio > 100:
                                    ratio = 100
                                p.grad += (ratio*(R-mean_R)+mean_D)
                            else:
                                match = (p.grad>0)*(R>0)
                                p.grad[match] += R[match]*0.001
                                p.grad[~match] *= .1
                                p.grad[~match] += R[~match]*0.0001
                        else:
                            raise NotImplementedError('Unknown gradient balance method: {}'.format(self.balance_loss))
                        assert(not torch.isnan(p.grad).any())
            self.saved_grads=[]
            for p in self.parameters:
                if p.grad is not None:
                    p.grad/=sum_multipliers
        #for p in self.model.parameters():
        #    if p.grad is not None:
        #        assert(not torch.isnan(p.grad).any())
        #        p.grad[torch.isnan(p.grad)]=0

        if 'no-step' not in lesson:
            torch.nn.utils.clip_grad_value_(self.model.parameters(),2) #prevent huge gradients
            for m in self.model.parameters():
                assert(not torch.isnan(m).any())

            if 'disc' in lesson or 'auto-disc' in lesson or 'disc-style' in lesson or 'author-train' in lesson or 'disc_reg' in lesson:
                self.optimizer_discriminator.step()
            elif  any(['auto-style' in l for l in lesson]):
                self.optimizer_gen_only.step()
            else:
                self.optimizer.step()

        #assert(not torch.isnan(self.model.spacer.std).any())
        #for p in self.parameters:
        #    assert(not torch.isnan(p).any())


        loss = loss_item

        #gt = instance['gt']
        #if pred is not None:
        #    cer,wer, pred_str = self.getCER(gt,pred)
        #else:
        #    cer=0
        #    wer=0

        #t#toc=timeit.default_timer()#t#
        #t#print('iteration: '+str(toc-tic))#t#

        #tic=timeit.default_timer()
        metrics={}


        log = {
            'loss': loss,
            **losses,
            #'pred_str': pred_str

            #'CER': cer,
            #'WER': wer,
            **run_log,
            #'meangrad': meangrad,

            **metrics,
        }

        #if iteration%10==0:
        #image=None
        #queryMask=None
        #targetBoxes=None
        #outputBoxes=None
        #outputOffsets=None
        #loss=None
        #torch.cuda.empty_cache()


        return log#
    def _minor_log(self, log):
        ls=''
        for key,val in log.items():
            ls += key
            if type(val) is float:
                number = '{:.6f}'.format(val)
                if number == '0.000000':
                    number = str(val)
                ls +=': {},\t'.format(number)
            else:
                ls +=': {},\t'.format(val)
        self.logger.info('Train '+ls)
        for  key,value in self.to_display.items():
            self.logger.info('{} : {}'.format(key,value))
        self.to_display={}

    def _valid_epoch(self):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()

        total_loss=0
        total_qrLoss=0
        total_autoLoss=0
        total_losses=defaultdict(lambda: 0)
        total_proper_QR=0
        print('validate')
        with torch.no_grad():
            losses = defaultdict(lambda: 0)
            for batch_idx, instance in enumerate(self.valid_data_loader):
                if not self.logged:
                    print('validate: {}/{}'.format(batch_idx,len(self.valid_data_loader)), end='\r')
                losses,run_log = self.run(instance,self.curriculum.getValid())
                pred=None
            
                for name in losses.keys():
                    #losses[name] *= self.lossWeights[name[:-4]]
                    total_loss += losses[name].item()*self.lossWeights[name[:-4]]
                    total_losses['val_'+name] += losses[name].item()
                total_proper_QR+=run_log['proper_QR']

                #if pred is not None:
                #    pred = pred.detach().cpu().numpy()
                #    gt = instance['gt']
                #    cer,wer,_ = self.getCER(gt,pred)
                #    total_cer += cer
                #    total_wer += wer
        
        for name in total_losses.keys():
            total_losses[name]/=len(self.valid_data_loader)
        toRet={
                'val_loss': total_loss/len(self.valid_data_loader),
                #'val_CER': total_cer/len(self.valid_data_loader),
                #'val_WER': total_wer/len(self.valid_data_loader),
                'val_proper_QR': total_proper_QR/len(self.valid_data_loader),
                **total_losses
                }
        return toRet

    def onehot(self,label):
        label_onehot = torch.zeros(label.size(0),label.size(1),self.num_class)
        label_onehot.scatter_(2,label.cpu().view(label.size(0),label.size(1),1),1)
        #un-tensorized
        #for i in range(label.size(0)):
        #    for j in range(label.size(1)):
        #        label_onehot[i,j,label[i,j]]=1
        return label_onehot.to(label.device)



    def run(self,instance,lesson,get=[]):
        qr_image, targetvalid,targetchars = self._to_tensor(instance)
        batch_size = qr_image.size(0)

        losses = {}

        if 'gen_reg' in lesson:
            path_batch_size = max(1, batch_size // self.path_batch_shrink)
            qr_image = qr_image[:path_batch_size]
            noise = mixing_noise(path_batch_size, self.model.style_dim, self.mixing, qr_image.device)
            gen_image, latents = self.model(qr_image,noise,return_latent=True)
            if self.sample_disc and 'eval' not in lesson and 'valid' not in lesson:
                self.add_gen_sample(gen_image)
        elif 'gen' in lesson or 'disc' in lesson or 'gen' in get:
            noise = mixing_noise(batch_size, self.model.style_dim, self.mixing, qr_image.device)
            gen_image = self.model(qr_image,noise) #TODO

            if self.sample_disc and 'eval' not in lesson and 'valid' not in lesson:
                self.add_gen_sample(gen_image)
        else:
            gen_image = None

        if 'gen' in lesson and 'char' in self.loss and 'eval' not in lesson and 'skip-qr' not in lesson:
            gen_valid,gen_chars = self.model.qr_net(gen_image)
            losses['charLoss'] = self.loss['char'](gen_chars.reshape(batch_size*gen_chars.size(1),-1),targetchars.view(-1),*self.loss_params['char'])
            #assert(losses['charLoss']!=0)
            if losses['charLoss']<0.0001:
                del losses['charLoss']
            if 'valid' in self.loss:
                losses['validLoss'] = self.loss['valid'](gen_valid,targetvalid,*self.loss_params['valid'])
                if losses['validLoss']<0.0001:
                    del losses['validLoss']
            #gen_qrLoss = self.loss['QR'](gen_pred,label)
            #losses['QRLoss'] = gen_qrLoss



        #Get generated and real data to match sizes
        if 'sample-disc' in lesson or 'disc' in lesson or 'disc_reg' in lesson:
            real = instance['image']
            if self.with_cuda:
                real = real.to(self.gpu)
        if 'sample-disc' in lesson:

            fake = self.sample_gen(batch_size)
            if fake is None:
                return None,{}
            fake = fake.to(qr_image.device)
        else:
            fake = gen_image #could append normal QR images here

        if 'disc_reg' in lesson:
            real.requires_grad = True
            real_pred = self.model.discriminator(real)
            r1_loss = d_r1_loss(real_pred, real)
            losses['disc_regLoss'] = self.r1 / 2 * r1_loss * self.curriculum.d_reg_every + 0 * real_pred[0]

        if 'disc' in lesson or 'auto-disc' in lesson or 'sample-disc' in lesson:
            #WHERE DISCRIMINATOR LOSS IS COMPUTED
            #if fake.size(3)>real.size(3):
            #    diff = fake.size(3)-real.size(3)
            #    real = F.pad(real,(0,diff,0,0),'replicate')
            #elif fake.size(3)<real.size(3):
            #    diff = -(fake.size(3)-real.size(3))
            #    fake = F.pad(fake,(0,diff,0,0),'replicate')
            #    #real = real[:,:,:,:-diff]
            if self.StyleGAN2:
                fake_pred=self.model.discriminator(fake.detach())
                real_pred=self.model.discriminator(real)
                disc_loss = d_logistic_loss(real_pred, fake_pred)
            else:
                discriminator_pred = self.model.discriminator(torch.cat((real,fake),dim=0).detach(),True)
                if self.WGAN:
                    #Improved W-GAN
                    assert(len(discriminator_pred)==1)
                    disc_pred_real = discriminator_pred[0][:image.size(0)].mean()
                    disc_pred_fake = discriminator_pred[0][image.size(0):].mean()
                    ep = torch.empty(batch_size).uniform_()
                    ep = ep[:,None,None,None].expand(image.size(0),image.size(1),image.size(2),image.size(3)).cuda()
                    hatImgs = ep*image.detach() + (1-ep)*fake.detach()
                    hatImgs.requires_grad_(True)
                    disc_interpolates = self.model.discriminator(None,None,hatImgs)[0]
                    gradients = autograd.grad(outputs=disc_interpolates, inputs=hatImgs,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

                    gradients = gradients.view(gradients.size(0), -1)                              
                    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10#gp_lambda

                    disc_loss = disc_pred_fake - disc_pred_real + grad_penalty
                elif self.DCGAN:
                    disc_loss=0
                    for i in range(len(discriminator_pred)): #iterate over different disc losses
                        discriminator_pred_on_real = torch.sigmoid(discriminator_pred[i][:image.size(0)])
                        discriminator_pred_on_fake = torch.sigmoid(discriminator_pred[i][image.size(0):])
                        disc_loss += F.binary_cross_entropy(discriminator_pred_on_real,torch.ones_like(discriminator_pred_on_real)) +  F.binary_cross_entropy(discriminator_pred_on_fake,torch.zeros_like(discriminator_pred_on_fake))
                else:
                    #DEFAULT: hinge loss. Probably the best
                    disc_loss=0
                    for i in range(len(discriminator_pred)): #iterate over different disc losses (scales)
                        discriminator_pred_on_real = discriminator_pred[i][:real.size(0)]
                        discriminator_pred_on_fake = discriminator_pred[i][real.size(0):]
                        disc_loss += F.relu(1.0 - discriminator_pred_on_real).mean() + F.relu(1.0 + discriminator_pred_on_fake).mean()
                    disc_loss /= len(discriminator_pred)

            losses['discriminatorLoss']=disc_loss

        if 'gen_reg' in lesson:
            path_batch_size = max(1, batch_size // self.path_batch_shrink)
            path_loss, self.mean_path_length, path_lengths = g_path_regularize(
                    fake, latents, self.mean_path_length)
            weighted_path_loss = self.path_regularize * self.curriculum.g_reg_every * path_loss
            if self.path_batch_shrink:
               weighted_path_loss += 0 * fake[0, 0, 0, 0]
            losses['gen_regLoss'] = weighted_path_loss

        if ('gen' in lesson or 'auto-gen' in lesson) and 'eval' not in lesson:
            #WHERE GENERATOR LOSS IS COMPUTED
            gen_pred = self.model.discriminator(fake,False)
            gen_loss=0
            predicted_disc=[]
            if self.DCGAN:
                for gp in gen_pred:
                    gp=torch.sigmoid(gp)
                    gen_loss = F.binary_cross_entropy(gp,torch.ones_like(gp))
                    if 'disc' in get:
                        predicted_disc.append(gp.mean(dim=1).detach().cpu())
            elif self.StyleGAN2:
                losses['generatorLoss']=g_nonsaturating_loss(gen_pred)
            else:
                for gp in gen_pred: #scales (i believe)
                    gen_loss -= gp.mean()
                    if 'disc' in get:
                        if len(gp.size())>1:
                            predicted_disc.append(gp.mean(dim=1).detach().cpu())
                        else:
                            predicted_disc.append(gp.detach().cpu())
                gen_loss/=len(gen_pred)
                losses['generatorLoss']=gen_loss
        else:
            predicted_disc=None


        if ('gen' in lesson or 'auto-gen' in lesson or 'valid' in lesson or 'eval' in lesson):
            correctly_decoded=0

            for b in range(batch_size):
                read = util.zbar_decode(((gen_image[b]+1)*255/2).cpu().detach().permute(1,2,0).numpy().astype(np.uint8))
                #qqq = ((qr_image[b]+1)*255/2).cpu().permute(1,2,0).numpy()
                #readA = util.zbar_decode(qqq)
                #assert(readA==instance['gt_char'][b])
                #import pdb;pdb.set_trace()
                if read==instance['gt_char'][b]:
                    correctly_decoded+=1
                elif self.i_cant:
                    outim = ((gen_image[b]+1)*255/2).cpu().detach().permute(1,2,0).numpy().astype(np.uint8)
                    img_f.imwrite('i_cant/cant{}.png'.format(random.randrange(0,100)),outim)
                #else:
                #    print('read:{} | gt:{}'.format(read,instance['gt_char'][b]))
                proper_ratio = correctly_decoded/batch_size
                if proper_ratio ==1:
                    self.i_cant=True
            log={'proper_QR':proper_ratio}
            if 'valid' not in lesson and 'eval' not in lesson:
                if self.modulate_pixel_loss=='momentum' and self.iteration>self.modulate_pixel_loss_start:
                    if proper_ratio>=self.proper_accept:
                        self.pixel_weight_delta = (1-self.pixel_momentum_B_good)*(self.proper_accept-proper_ratio) + self.pixel_momentum_B_good*self.pixel_weight_delta
                        self.pixel_thresh_delta = (1-self.pixel_momentum_B_good)*(self.proper_accept-proper_ratio) + self.pixel_momentum_B_good*self.pixel_thresh_delta
                    else:
                        self.pixel_weight_delta = (1-self.pixel_momentum_B_bad)*(self.proper_accept-proper_ratio) + self.pixel_momentum_B_bad*self.pixel_weight_delta
                        self.pixel_thresh_delta = (1-self.pixel_momentum_B_bad)*(self.proper_accept-proper_ratio) + self.pixel_momentum_B_bad*self.pixel_thresh_delta
                    
                    init_weight=self.lossWeights['pixel']
                    self.lossWeights['pixel'] += self.pixel_weight_delta*self.pixel_weight_rate
                    self.lossWeights['pixel'] = min(max(self.lossWeights['pixel'],self.min_pixel_weight),self.max_pixel_weight)

                    if 'threshold' in self.loss_params['pixel']:
                        threshold = self.loss_params['pixel']['threshold']
                    else:
                        threshold = self.loss['pixel'].threshold
                    threshold -= self.pixel_thresh_delta*self.pixel_thresh_rate
                    threshold = min(max(threshold,0.05),1.0)
                    if 'threshold' in self.loss_params['pixel']:
                        self.loss_params['pixel']['threshold'] = threshold
                    else:
                        self.loss['pixel'].threshold = threshold
                    #This here is my hack method of allowing the training to resume at the same weight and thresh
                    self.config['loss_weights']['pixel']=self.lossWeights['pixel']
                    self.config['loss_params']['pixel']['threshold']=threshold
                    if init_weight != self.lossWeights['pixel']:
                        print('proper:{}  pixel loss weight:{:.4} D({:.4}), threshold:{:.4} D({:.4})'.format(proper_ratio,self.lossWeights['pixel'],self.pixel_weight_delta,threshold,self.pixel_thresh_delta))
                #elif self.modulate_pixel_loss=='bang':


        else:
            log={}

        if ('gen' in lesson or 'auto-gen' in lesson) and 'pixel' in self.loss and 'skip-pixel' not in lesson:
            if self.modulate_pixel_loss!='bang' or proper_ratio>self.proper_accept or self.iteration<self.modulate_pixel_loss_start:
                losses['pixelLoss'] = self.loss['pixel'](gen_image,qr_image,**self.loss_params['pixel'])

        if get:
            if (len(get)>1 or get[0]=='style') and 'name' in instance:
                got={'name': instance['name']}
            else:
                got={}
            for name in get:
                if name=='gen_image':
                    got[name] = gen_image.cpu().detach()
                elif name=='gen_img':
                    got[name] = gen_image.cpu().detach()
                elif name=='gen':
                    if gen_image is not None:
                        got[name] = gen_image.cpu().detach()
                    else:
                        print('ERROR, gen_image is None, lesson: {}'.format(lesson))
                        #got[name] = None
                elif name=='pred':
                    if self.model.pred is None:
                        self.model.pred = self.model.hwr(image, None)   
                    got[name] = self.model.pred.cpu().detach()
                elif name=='gt':
                    got[name] = instance['gt']
                elif name=='disc':
                    got[name] = predicted_disc
                else:
                    raise ValueError("Unknown get [{}]".format(name))
            ret = (losses, log, got)
        else:
            ret = (losses, log)
        #self.model.spaced_label=None
        #self.model.mask=None
        #self.model.gen_mask=None
        #self.model.top_and_bottom=None
        #self.model.counts=None
        #self.model.pred=None
        #self.model.spacing_pred=None
        #self.model.mask_pred=None
        #self.model.gen_spaced=None
        #self.model.spaced_style=None
        #self.model.mu=None
        #self.model.sigma=None
        return ret

    def getCER(self,gt,pred,individual=False):
        cer=0
        wer=0
        if individual:
            all_cer=[]
        pred_strs=[]
        for i,gt_line in enumerate(gt):
            logits = pred[:,i]
            pred_str, raw_pred = string_utils.naive_decode(logits)
            pred_str = string_utils.label2str_single(pred_str, self.idx_to_char, False)
            this_cer = error_rates.cer(gt_line, pred_str, self.casesensitive)
            cer+=this_cer
            if individual:
                all_cer.append(this_cer)
            pred_strs.append(pred_str)
            wer += error_rates.wer(gt_line, pred_str, self.casesensitive)
        cer/=len(gt)
        wer/=len(gt)
        if individual:
            return cer,wer, pred_strs, all_cer
        return cer,wer, pred_strs

    def sample_gen(self,batch_size):
        images=[]
        labels=[]

        #max_w=0
        #max_l=0
        for b in range(batch_size):
            if (random.random()<self.new_gen_freq or len(self.old_gen)<10) and len(self.new_gen)>0:
                #new
                inst = self.new_gen[0]
                self.new_gen = self.new_gen[1:]
            elif  len(self.old_gen)>0:
                i = random.randint(0,len(self.old_gen)-1)
                if self.old_gen_cache is not None:
                    inst = torch.load(self.old_gen[i])
                else:
                    inst = self.old_gen[i]
            else:
                return None

            if type(inst) is tuple:
                image,label = inst
                labels.append(label)
                #max_l = max(max_l,label.size(0))
            else:
                image = inst
                label=None
            images.append(image)
            #max_w = max(max_w,image.size(3))
            #if label is not None:
        #for b in range(batch_size):
        #    if images[b].size(3)<max_w:
        #        diff = max_w -  images[b].size(3)
        #        images[b] = F.pad( images[b], (0,diff),value=PADDING_CONSTANT)
        #    if labels[b].size(0)<max_l:
        #        diff = max_l -  labels[b].size(0)
        #        labels[b] = F.pad( labels[b].permute(1,2,0), (0,diff),value=PADDING_CONSTANT).permute(2,0,1)
        
        assert(len(images)==batch_size)
        if len(labels)>0:
            return torch.cat(images,dim=0), torch.cat(labels,dim=0)
        else:
            return torch.cat(images,dim=0)

    def add_gen_sample(self,images,labels=None):
        batch_size = images.size(0)
        images = images.cpu().detach()
        if labels is not None:
            labels = labels.cpu().detach()

        for b in range(batch_size):
            if labels is not None:
                inst = (images[b:b+1],labels[:,b:b+1],styles[b:b+1])
            else:
                inst = images[b:b+1]
            if len(self.new_gen)>= self.store_new_gen_limit:
                old = self.new_gen[0]
                self.new_gen = self.new_gen[1:]+[inst]

                if len(self.old_gen)>= self.store_old_gen_limit:
                    if random.random() > self.forget_new_freq:
                        change = random.randint(0,len(self.old_gen)-1)
                        if self.old_gen_cache is not None:
                            torch.save(old,self.old_gen[change])
                        else:
                            self.old_gen[change] = old
                else:
                    if self.old_gen_cache is not None:
                        path = os.path.join(self.old_gen_cache,'{}.pt'.format(len(self.old_gen)))
                        torch.save(old,path)
                        self.old_gen.append(path)
                    else:
                        self.old_gen.append(old)
            else:
                self.new_gen.append(inst)


    def print_images(self,images,text,disc=None,typ='gen',gtImages=None):
        if self.print_dir is not None:
            images = images.detach()
            nrow = max(1,2048//images.size(3))
            if self.iteration-self.last_print_images[typ]>=self.serperate_print_every:
                iterP = self.iteration
                self.last_print_images[typ]=self.iteration
            else:
                iterP = 'latest'
            vutils.save_image(images,
                    os.path.join(self.print_dir,'{}_samples_{}.png'.format(typ,iterP)),
                    nrow=nrow,
                    normalize=True)
            if gtImages is not None:
                gtImages = gtImages.detach()
                vutils.save_image(gtImages,
                        os.path.join(self.print_dir,'{}_gt_{}.png'.format(typ,iterP)),
                        nrow=nrow,
                        normalize=True)
            if typ=='gen': #reduce clutter, GT should be visible from GT image
                with open(os.path.join(self.print_dir,'{}_text_{}.txt'.format(typ,iterP)),'w') as f:
                    if disc is None or len(disc)==0:
                        f.write('\n'.join(text))
                    else:
                        for i,t in enumerate(text):
                            f.write(t)
                            for v in disc:
                                if i < v.size(0):
                                    f.write(', {}'.format(v[i].mean().item()))
                            f.write('\n')
            print('printed {} images, iter: {}'.format(typ,self.iteration))

