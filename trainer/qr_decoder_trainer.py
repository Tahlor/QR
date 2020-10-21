import numpy as np
import torch
from base import BaseTrainer
import timeit
from utils.error_rates import cer


class QRDecoderTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        self.optimizer is by default handled by BaseTrainer based on config.
    """
    def __init__(self, model, loss, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(QRDecoderTrainer, self).__init__(model, loss, metrics, resume, config, train_logger)
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.data_loader_iter = iter(data_loader)
        #for i in range(self.start_iteration,
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False
        #self.log_step = int(np.sqrt(self.batch_size))

        if 'loss_params' in config:
            self.loss_params=config['loss_params']
        else:
            self.loss_params={}
	self.lossWeights = config['loss_weights'] if 'loss_weights' in config else defaultdict(lambda:1)

    #def _to_tensor(self, data, target):
    #    return self._to_tensor_individual(data), _to_tensor_individual(target)
    def _to_tensor(self, *datas):
        ret=(self._to_tensor_individual(datas[0]),)
        for i in range(1,len(datas)):
            ret+=(self._to_tensor_individual(datas[i]),)
        return ret
    def _to_tensor_individual(self, data):
        if type(data)==str:
            return data
        if type(data)==list or type(data)==tuple:
            return [self._to_tensor_individual(d) for d in data]
        if (len(data.size())==1 and data.size(0)==1):
            return data[0]
        if type(data) is np.ndarray:
            data = torch.FloatTensor(data.astype(np.float32))
        #elif type(data) is torch.Tensor:
        #    data = data.type(torch.FloatTensor)
        if self.with_cuda:
            data = data.to(self.gpu)
        return data

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        if len(self.metrics)>0:
            output = output.cpu().data.numpy()
            target = target.cpu().data.numpy()
            for i, metric in enumerate(self.metrics):
                acc_metrics[i] += metric(output, target)
        return acc_metrics

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
        self.optimizer.zero_grad()

        #tic=timeit.default_timer()
        #batch_idx = (iteration-1) % len(self.data_loader)
        try:
            losses,run_log,out = self.run(self.data_loader_iter.next())
            
            #data, target = self._to_tensor(*self.data_loader_iter.next())
        except StopIteration:
            self.data_loader_iter = iter(self.data_loader)
            losses,run_log,out = self.run(self.data_loader_iter.next())
            #data, target = self._to_tensor(*self.data_loader_iter.next())
        #toc=timeit.default_timer()
        #print('data: '+str(toc-tic))
        
        #tic=timeit.default_timer()
	loss=0
        for name in losses.keys():
            losses[name] *= self.lossWeights[name[:-4]]
            loss += losses[name]
            losses[name] = losses[name].item()
        if len(losses)>0:
            loss.backward()
        self.optimizer.step()

        #toc=timeit.default_timer()
        #print('for/bac: '+str(toc-tic))

        #tic=timeit.default_timer()
        metrics = self._eval_metrics(output, target)
        #toc=timeit.default_timer()
        #print('metric: '+str(toc-tic))

        #tic=timeit.default_timer()
        loss = loss.item()
        #toc=timeit.default_timer()
        #print('item: '+str(toc-tic))


        log = {
            'loss': loss,
            'metrics': metrics,
           **losses,
           **run_log
        }


        return log

    def _minor_log(self, log):
        ls=''
        for key,val in log.items():
            ls += key
            if type(val) is float:
                ls +=': {:.6f}, '.format(val)
            else:
                ls +=': {}, '.format(val)
        self.logger.info('Train '+ls)

    def _valid_epoch(self):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_loss = 0
        total_losses=defaultdict(lambda: 0)
        total_cer=0
        total_valid_acc=0
        #total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, instance in enumerate(self.valid_data_loader):
                if not self.logged:
                    print('validate: {}/{}'.format(batch_idx,len(self.valid_data_loader)), end='\r')
                losses,log,out = self.run(instance)
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)

		for name in losses.keys():
                    #losses[name] *= self.lossWeights[name[:-4]]
                    total_loss += losses[name].item()*self.lossWeights[name[:-4]]
                    total_losses['val_'+name] += losses[name].item()
                total_cer+=log['cer']
                total_valid_acc+=log['valid_acc']


        for name in total_losses.keys():
            total_losses[name]/=len(self.valid_data_loader)

        return {
            'val_loss': total_loss / len(self.valid_data_loader),
            'val_cer': total_cer / len(self.valid_data_loader),
            'val_valid_acc': total_valid_acc / len(self.valid_data_loader),
            **total_losses
            #'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }

    def run(self,instance):
        data,targetvalid,targetchars = self._to_tensor(instance)
        gt_chars = instance['?']
        outvalid, outchars = self.model(data)
        losses={}
        loss['char'] = self.loss['char'](outchars,targetchars,*self.lossParams['char'])
        loss['valid'] = self.loss['valid'](outvalids,targetvalids,*self.lossParams['valid'])

        chars=[]
        char_indexes = outchars.argmax(dim=2)
        batch_size = outchars.size(0)
        for b in range(batch_size):
            s=''
            for p in range(outchars.size(1)):
                s+=self.index_to_char[char_indexes[b,p]]
            chars+=s
            
            b_cer += cer(s,gt_chars[b])
            
        acc = torch.logical_and(outvalid[b]>0,targetvalid[b]>0).mean()
        log={
                'cer':b_cer/batch_size,
                'valid_acc':acc
                }
                

        return losses,log,chars
