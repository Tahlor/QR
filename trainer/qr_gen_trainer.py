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
        if 'loss_params' in config:
            self.loss_params=config['loss_params']
        else:
            self.loss_params={}
        for lossname in self.loss:
            if lossname not in self.loss_params:
                self.loss_params[lossname]={}
        self.lossWeights = config['loss_weights'] if 'loss_weights' in config else {"auto": 1, "recog": 1}
        if data_loader is not None:
            self.batch_size = data_loader.batch_size
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
            self.parameters = list(model.parameters())
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



        if 'alt_data_loader' in config:
            alt_config={'data_loader': config['alt_data_loader'],'validation':{}}
            self.alt_data_loader, alt_valid_data_loader = getDataLoader(alt_config,'train')
            self.alt_data_loader_iter = iter(self.alt_data_loader)
        if 'triplet_data_loader' in config:
            triplet_config={'data_loader': config['triplet_data_loader'],'validation':{}}
            self.triplet_data_loader, triplet_valid_data_loader = getDataLoader(triplet_config,'train')
            self.triplet_data_loader_iter = iter(self.triplet_data_loader)


    def _to_tensor(self, instance):
        image = instance['image']
        label = instance['label']

        if self.with_cuda:
            if image is not None:
                image = image.to(self.gpu)
            if label is not None:
                label = label.to(self.gpu)
        return image, label

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

        ##tic=timeit.default_timer()
        lesson =  self.curriculum.getLesson(iteration)
        if all([l[:3]=='gen' or l=='no-step' for l in lesson]) and self.text_data is not None:
            instance = self.text_data.getInstance()
        else:
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
            losses,got = self.run(instance,lesson,get=['gen'])
            self.print_images(got['gen'],instance['gt_text'],gtImages=instance['image'],typ='gen')
            self.iter_to_print=self.print_every
        else:
            losses = self.run(instance,lesson)
            self.iter_to_print-=1
        pred=None

        if losses is None:
            return {}
        #TODO rewrite these for the losses we need to balance (probably just QR reading and validation)
        loss=0
        qrLoss=0
        #authorClassLoss=0
        #autoGenLoss=0
        for name in losses.keys():
            losses[name] *= self.lossWeights[name[:-4]]
            #if self.balance_loss and 'generator' in name and 'auto-gen' in lesson:
            #    autoGenLoss += losses[name]
            #elif self.balance_loss and 'Recog' in name:
            #    qrLoss += losses[name]
            #elif self.balance_loss and 'authorClass' in name:
            #    authorClassLoss += losses[name]
            if self.balance_loss and 'QR' in name:
                qrLoss += losses[name]
            else:
                loss += losses[name]
            losses[name] = losses[name].item()
        losses_to_balance = [qrLoss]
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
                        if p.grad is None:
                            saved_grad.append(None)
                        else:
                            saved_grad.append(p.grad.clone())
                            p.grad.zero_()
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
                for iterT,mult in self.balance_var_x.items():
                    if int(iterT)<=iteration:
                        multipliers=mult
                        if type(multipliers) is not list:
                            multipliers=[multipliers]
            for gi,saved_grad in enumerate(self.saved_grads):
                if self.balance_loss.startswith('sign_preserve_var'):
                    x=multipliers[gi]
                for i,(R, p) in enumerate(zip(saved_grad, self.parameters)):
                    if R is not None:
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
                        elif self.balance_loss=='sign_preserve_moreHWR':
                            abmean_R = torch.abs(R).mean()
                            if abmean_R!=0:
                                p.grad += 2*R*(abmean_Ds[i]/abmean_R)
                        elif self.balance_loss.startswith('sign_preserve_var'): #This is the best, as you can specify a weight for each balanced term
                            abmean_R = torch.abs(R).mean()
                            if abmean_R!=0:
                                p.grad += x*R*(abmean_Ds[i]/abmean_R)
                        elif self.balance_loss.startswith('sign_preserve_x'):
                            abmean_R = torch.abs(R).mean()
                            if abmean_R!=0:
                                p.grad += self.balance_x*R*(abmean_Ds[i]/abmean_R)
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
        #for p in self.model.parameters():
        #    if p.grad is not None:
        #        assert(not torch.isnan(p.grad).any())
        #        p.grad[torch.isnan(p.grad)]=0

        if 'no-step' not in lesson:
            torch.nn.utils.clip_grad_value_(self.model.parameters(),2) #prevent huge gradients
            for m in self.model.parameters():
                assert(not torch.isnan(m).any())

            if 'disc' in lesson or 'auto-disc' in lesson or 'disc-style' in lesson or 'author-train' in lesson:
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

        ##toc=timeit.default_timer()
        ##print('bac: '+str(toc-tic))

        #tic=timeit.default_timer()
        metrics={}


        log = {
            'loss': loss,
            **losses,
            #'pred_str': pred_str

            #'CER': cer,
            #'WER': wer,
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
        total_cer=0
        total_wer=0
        print('validate')
        with torch.no_grad():
            losses = defaultdict(lambda: 0)
            for batch_idx, instance in enumerate(self.valid_data_loader):
                if not self.logged:
                    print('validate: {}/{}'.format(batch_idx,len(self.valid_data_loader)), end='\r')
                losses = self.run(instance,self.curriculum.getValid())
                pred=None
            
                for name in losses.keys():
                    #losses[name] *= self.lossWeights[name[:-4]]
                    total_loss += losses[name].item()*self.lossWeights[name[:-4]]
                    total_losses['val_'+name] += losses[name].item()

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
        image, label = self._to_tensor(instance)
        batch_size = label.size(1)
        label_lengths = instance['label_lengths']

        losses = {}

        if 'gen' in lesson or 'disc' in lesson or 'gen' in get:
            gen_image = self.model(image) #TODO

            if self.sample_disc and 'eval' not in lesson and 'valid' not in lesson:
                self.add_gen_sample(gen_image)
        else:
            gen_image = None

        if 'gen' in lesson and 'QR' in self.loss and 'eval' not in lesson:
            gen_pred = self.model.qr_net(gen_image)
            gen_qrLoss = self.loss['QR'](gen_pred,label)
            losses['QRLoss'] = gen_qrLoss



        #Get generated and real data to match sizes
        if 'disc' in lesson:
            fake = gen_image #could append normal QR images here
            real = image
        elif 'sample-disc' in lesson:
            real = image

            fake = self.sample_gen(batch_size)
            if fake is None:
                return None
            fake = fake.to(image.device)

        if 'disc' in lesson or 'auto-disc' in lesson or 'sample-disc' in lesson:
            #WHERE DISCRIMINATOR LOSS IS COMPUTED
            if fake.size(3)>image.size(3):
                diff = fake.size(3)-image.size(3)
                image = F.pad(image,(0,diff,0,0),'replicate')
            elif fake.size(3)<image.size(3):
                diff = -(fake.size(3)-image.size(3))
                fake = F.pad(fake,(0,diff,0,0),'replicate')
                #image = image[:,:,:,:-diff]
            ##DEBUG
            #for i in range(batch_size):
            #    im = ((1-image[i,0])*127).cpu().numpy().astype(np.uint8)
            #    cv2.imwrite('test/real{}.png'.format(i),im)
            #for i in range(batch_size):
            #    im = ((1-fake[i,0])*127).cpu().numpy().astype(np.uint8)
            #    cv2.imwrite('test/fake{}.png'.format(i),im)
            #print(lesson)
            #import pdb;pdb.set_trace()

            discriminator_pred = self.model.discriminator(torch.cat((real,fake),dim=0).detach())
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

        if ('gen' in lesson or 'auto-gen' in lesson) and 'eval' not in lesson:
            #WHERE GENERATOR LOSS IS COMPUTED
            gen_pred = self.model.discriminator(fake)
            gen_loss=0
            predicted_disc=[]
            if self.DCGAN:
                for gp in gen_pred:
                    gp=torch.sigmoid(gp)
                    gen_loss = F.binary_cross_entropy(gp,torch.ones_like(gp))
                    if 'disc' in get:
                        predicted_disc.append(gp.mean(dim=1).detach().cpu())
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
            ret = (losses, got)
        else:
            ret = losses
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

