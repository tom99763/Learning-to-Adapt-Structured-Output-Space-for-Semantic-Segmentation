import argparse
from utils import *
from models import AdaptFCN
from tensorflow.keras import optimizers

def parse_opt():
    parser = argparse.ArgumentParser()
    #dataset & dirs
    parser.add_argument('--source_pth', type=str, default='./datasets/crack/source')
    parser.add_argument('--target_pth', type=str, default='./datasets/crack/target')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
    parser.add_argument('--result_dir', type=str, default='./results')
    parser.add_argument('--val_size', type=str, default=0.2)
    parser.add_argument('--seed', type=str, default=999)
    parser.add_argument('--batch_size', type=str, default=16)
    parser.add_argument('--img_size', type=str, default=224)
    parser.add_argument('--num_samples', type=str, default=4)
    
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
    parser.add_argument('--num_epochs', type=str, default=100)
    opt, _ = parser.parse_known_args()
    return opt


def main(opt):
    ds_train, ds_val = create_dataset(opt)
    adaptfcn = AdaptFCN.AdaptFCN(opt)
    adaptfcn.compile(optimizer = [
        optimizers.Adam(learning_rate=opt.lr, beta_1=opt.beta_1, beta_2=opt.beta_2),
        optimizers.Adam(learning_rate=opt.lr, beta_1=opt.beta_1, beta_2=opt.beta_2)
    ])
    
    sample= next(iter(ds_val.take(1)))
    callbacks = create_callbacks(opt, sample)
    
    adaptfcn.fit(
        x=ds_train,
        validation_data=ds_val,
        epochs=opt.num_epochs,
        callbacks=callbacks
    )

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
