from .decoder_eval import Decoder_eval

def SimpleQRDataset_eval(config,instance, trainer, metrics, outDir=None, startIndex=None, lossFunc=None,toEval=None):
    return Decoder_eval(config,instance, trainer, metrics, outDir, startIndex, lossFunc, toEval)
