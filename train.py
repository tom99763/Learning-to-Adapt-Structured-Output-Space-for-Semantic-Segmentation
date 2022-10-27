import argparse
from utils import *
from models import AdaptFCN

def parse_opt():
    parser = argparse.ArgumentParser()
    #dataset
    parser.add_argument('--source_pth', type=str, default='../dataset/crack/source')
    parser.add_argument('--target_pth', type=str, default='../dataset/crack/target')
    parser.add_argument('--val_size', type=str, default=0.2)
    parser.add_argument('--seed', type=str, default=999)
    parser.add_argument('--batch_size', type=str, default=32)
    parser.add_argument('--img_size', type=str, default=224)
    
    #network
    parser.add_argument('--base', type=str, default=64)
    parser.add_argument('--num_classes', type=str, default=2)
    parser.add_argument('--num_downsamples', type=str, default=4)
    opt, _ = parser.parse_known_args()
    return opt


def main(opt):
    ds_train, ds_val = create_dataset(opt)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
