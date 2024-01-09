from torch.utils import data
from torch.nn.parallel import DistributedDataParallel as DDP
from easydict import EasyDict as edict
from pathlib import Path
import utils.misc as misc
import torch
import os
from progress.bar import Bar


def test(model, test_loaders, test_json_list, ckpt_dir, args, epoch=0):
    model.eval()
    # metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 20
    log_perf = ''
    gpsm = []
    for name, loader in zip(test_json_list, test_loaders):
        bar = Bar(f'Validation Epoch {epoch + 1} on COCO2014', fill='#', max=len(loader))
        for _, samples in enumerate(loader):
            with torch.no_grad():
                for k, v in samples.items():
                    if isinstance(v, torch.Tensor):
                        samples[k] = samples[k].to('cuda')
                if args.fp32:
                    model(**samples, mode='eval')
                else:
                    with torch.cuda.amp.autocast():
                        model(**samples, mode='eval')
        gpsm.append(model.loss.accumulate())
        log_perf += f'{name} {gpsm[-1]}\n'

    gpsm = sum([val * weight for val, weight in zip(gpsm, args.dataset_weights)])
    log_perf += f'avg {gpsm}\n'
    print(f'average gpsm on COCO2014 is {gpsm}')
    '''
    if torch.distributed.get_rank() == 0:
        with open(os.path.join(ckpt_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(log_perf)
        print(log_perf)
    '''
