import argparse
from utils import *

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_pth', type=str, default='../input/crack-dataset/crack/source')
    parser.add_argument('--target_pth', type=str, default='../input/crack-dataset/crack/target')
    parser.add_argument('--val_size', type=str, default=0.2)
    parser.add_argument('--seed', type=str, default=999)
    parser.add_argument('--batch_size', type=str, default=32)
    parser.add_argument('--img_size', type=str, default=224)
    opt, _ = parser.parse_known_args()
    return opt


def main(opt):
    ds_train, ds_val = create_dataset(opt)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
