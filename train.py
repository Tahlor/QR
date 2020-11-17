import torch
import os
import sys
import signal
import json
import logging
import argparse
from model import *
from model.loss import *
from model.metric import *
from data_loader import getDataLoader
from trainer import *
from logger import Logger

def update_status(name,message,supercomputer):
    if supercomputer:
        return
    import requests
    import socket
    name = '{}: {}'.format(socket.gethostname(),name)
    try:
        proxies = {
                  "http": None,
                    "https": None,
                    }
        #print('http://sensei-status.herokuapp.com/sensei-update/{}?message={}'.format(name,message))
        r = requests.get('http://sensei-status.herokuapp.com/sensei-update/{}?message={}'.format(name,message),proxies=proxies)
    except requests.exceptions.ConnectionError:
        pass
    pass


logging.basicConfig(level=logging.INFO, format='')
def set_procname(newname):
        from ctypes import cdll, byref, create_string_buffer
        newname=os.fsencode(newname)
        libc = cdll.LoadLibrary('libc.so.6')    #Loading a 3rd party library C
        buff = create_string_buffer(len(newname)+1) #Note: One larger than the name (man prctl says that)
        buff.value = newname                 #Null terminated string as it should be
        libc.prctl(15, byref(buff), 0, 0, 0) #Refer to "#define" of "/usr/include/linux/prctl.h" for the misterious value 16 & arg[3..5] are zero as the man page says.

def main(config, resume):
    supercomputer = config['super_computer'] if 'super_computer' in config else False
    #set_procname(config['name'])
    #np.random.seed(1234) I don't have a way of restarting the DataLoader at the same place, so this makes it totaly random
    train_logger = Logger()

    split = config['split'] if 'split' in config else 'train'
    data_loader, valid_data_loader = getDataLoader(config,split)
    #valid_data_loader = data_loader.split_validation()
    model = eval(config['model']['arch'])(config['model'])
    model.summary()
    if config['trainer']['class']=='HWRWithSynthTrainer':
        gen_model = model
        model = model.hwr
        gen_model.hwr=None
        #config['gen_model$'] = gen_model
    if type(config['loss'])==dict:
        loss={}#[eval(l) for l in config['loss']]
        for name,l in config['loss'].items():
            loss[name]=eval(l)
    else:
        loss = eval(config['loss'])
    if type(config['metrics'])==dict:
        metrics={}
        for name,m in config['metrics'].items():
            metrics[name]=[eval(metric) for metric in m]
    else:
        metrics = [eval(metric) for metric in config['metrics']]

    if 'class' in config['trainer']:
        trainerClass = eval(config['trainer']['class'])
    else:
        trainerClass = Trainer
    print('before trainer')
    trainer = trainerClass(model, loss, metrics,
                      resume=resume,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      train_logger=train_logger)
    if config['trainer']['class']=='HWRWithSynthTrainer':
        trainer.gen = gen_model
    print('after trainer')

    name=config['name']
    def handleSIGINT(sig, frame):
        trainer.save()
        update_status(name,'stopped!',supercomputer)
        sys.exit(0)
    signal.signal(signal.SIGINT, handleSIGINT)

    print("Begin training")
    update_status(name,'started',supercomputer)
    trainer.train()


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-s', '--soft_resume', default=None, type=str,
                        help='path to checkpoint that may or may not exist (default: None)')
    parser.add_argument('-g', '--gpu', default=None, type=int,
                        help='gpu to use (overrides config) (default: None)')
    parser.add_argument('-p', '--ping', default=False, action='store_const', const=True,
                        help='should send updates to my ping server')

    args = parser.parse_args()

    config = None
    if args.config is not None:
        config = json.load(open(args.config))
    if  args.resume is None and  args.soft_resume is not None:
        if not os.path.exists(args.soft_resume):
            print('WARNING: resume path ({}) was not found, starting from scratch'.format(args.soft_resume))
        else:
            args.resume = args.soft_resume
    elif args.resume is not None and (config is None or 'override' not in config or not config['override']):
        if args.config is not None:
            logger.warning('Warning: --config overridden by --resume')
        config = torch.load(args.resume)['config']
    elif args.config is not None and args.resume is None:
        path = os.path.join(config['trainer']['save_dir'], config['name'])
        if os.path.exists(path):
            directory = os.fsencode(path)
            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                if 'checkpoint' in filename: 
                    assert False, "Path {} already used!".format(path)
    assert config is not None
    supercomputer = config['super_computer'] if 'super_computer' in config else False
    supercomputer = supercomputer or args.ping

    name=config['name']
    file_name = args.config
    if '/' in file_name:
        file_name = file_name[file_name.rindex('/')+4:-5] #remove path
    else:
        file_name = file_name[3:-5]
    if name!=file_name:
        raise Exception('ERROR, name and file name do not match, {} != {} ({})'.format(name,file_name,args.config))
    

    if args.gpu is not None:
        config['gpu']=args.gpu
        print('override gpu to '+str(config['gpu']))
    try: 
        if config['cuda']:
            with torch.cuda.device(config['gpu']):
                main(config, args.resume)
        else:
            main(config, args.resume)
    except Exception as er:
        name=config['name']

        #urllib.request.urlopen(url
        update_status(name,er,supercomputer)
        raise er
    else:
        name=config['name']
        update_status(name,'DONE!',supercomputer)
