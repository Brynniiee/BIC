import torch
import numpy as np
from trainer import Trainer
import sys
from utils import *
import argparse

parser = argparse.ArgumentParser(description='Incremental Learning BIC')
parser.add_argument('--batch_size', default = 18, type = int)
parser.add_argument('--epoch', default = 2, type = int) # small values for debug
parser.add_argument('--lr', default = 0.1, type = float) 
parser.add_argument('--max_size', default = 2000, type = int)
parser.add_argument('--init_cls', default = 0, type = int)  #### total_cls=100 -> init_cls=0
args = parser.parse_args()


if __name__ == "__main__":
    showGod()
    trainer = Trainer(args.init_cls)
    trainer.train(args.batch_size, args.epoch, args.lr, args.max_size)
