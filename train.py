import argparse
from utils import *
from models import AdaptFCN
from tensorflow.keras import optimizers

def parse_opt():
    parser = argparse.ArgumentParser()
    #dataset & dirs
    parser.add_argument('--source_pth', type=str, default='./dataset/crack/source')
    parser.add_argument('--target_pth', type=str, default='./dataset/crack/target')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
    parser.add_argument('--val_size', type=str, default=0.2)
    parser.add_argument('--seed', type=str, default=999)
    parser.add_argument('--batch_size', type=str, default=32)
    parser.add_argument('--img_size', type=str, default=224)
    
    #network
    parser.add_argument('--base', type=str, default=64)
    parser.add_argument('--num_classes', type=str, default=2)
    parser.add_argument('--num_downsamples', type=str, default=4)
    
    #loss
    parser.add_argument('--gan_loss', type=str, default='ls')
    parser.add_argument('--lambda_cls', type=str, default=1.)
    parser.add_argument('--lambda_adv', type=str, default=1.)
    
    #optimization
    parser.add_argument('--beta_1', type=str, default=0.9)
    parser.add_argument('--beta_2', type=str, default=0.99)
    parser.add_argument('--lr', type=str, default=1e-4)
    arser.add_argument('--num_epochs', type=str, default=100)
    opt, _ = parser.parse_known_args()
    return opt


def main(opt):
    ds_train, ds_val = create_dataset(opt)
    adaptfcn = Adaptfcn(opt)
    adaptfcn.compile(optimizer = [
        optimizers.Adam(learning_rate=opt.lr, beta_1=opt.beta_1, beta_2=opt.beta_2),
        optimizers.Adam(learning_rate=opt.lr, beta_1=opt.beta_1, beta_2=opt.beta_2)
    ])
    callbacks = create_callbacks(opt)
    
    adaptfcn.fit(
        x=ds_train,
        validation_data=ds_val,
        epochs=opt.num_epochs,
        callbacks
    )

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
