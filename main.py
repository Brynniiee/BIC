import torch
import numpy as np
from trainer import Trainer
import sys
import argparse

parser = argparse.ArgumentParser(description='Incremental Learning BIC')
parser.add_argument('--batch_size', default = 8, type = int)
parser.add_argument('--epoch', default = 1, type = int) # small values for debug
parser.add_argument('--lr', default = 0.01, type = float) 
parser.add_argument('--max_size', default = 100, type = int)
parser.add_argument('--init_cls', default = 0, type = int)  #### total_cls=100 -> init_cls=0
parser.add_argument('--distill_temperature', default = 1, type = float)
parser.add_argument('--bias_lr', default = 0.001, type = float)
parser.add_argument('--beta',default= 0.0, type = float)
parser.add_argument('--resume_task', default = 0, type = int)
args = parser.parse_args()



if __name__ == "__main__":
    trainer = Trainer(args.init_cls)
    trainer.train(args.batch_size, args.epoch, args.lr, args.bias_lr, args.max_size, T = args.distill_temperature, beta = args.beta, resume_task = args.resume_task)
