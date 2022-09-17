import argparse
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--data', default="./Datasets")
        self.parser.add_argument('--dataset', default="OCT")
        self.parser.add_argument('--dataset_Super', default="OCT_Super")
        self.parser.add_argument('--num_epochs', type=int, default=100)
        self.parser.add_argument('--log_dirs', type=str, default='./Log')
        self.parser.add_argument('--crop_height', type=int, default=512)
        self.parser.add_argument('--crop_width', type=int, default=256)
        self.parser.add_argument('--batch_size', type=int, default=8)
        self.parser.add_argument('--labeled_bs', type=int, default=4)
        self.parser.add_argument('--num_threads', default=4, type=int)
        self.parser.add_argument('--net_work', type=str, default='Capsule')
        self.parser.add_argument('--input_ch', type=int, default=1)
        self.parser.add_argument('--out_ch', type=int, default=1)
        self.parser.add_argument('--lr_mode', type=str, default="poly")
        self.parser.add_argument('--beta1', type=float, default=0.5)
        self.parser.add_argument('--lr', type=float, default=0.0002)
        self.parser.add_argument('--use_gpu', type=bool, default=True)
        self.parser.add_argument('--gpu_ids', type=str, default='0')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./Semi_Fused_Results')
        self.parser.add_argument('--netG', type=str, default='Capsule')
        self.parser.add_argument('--cuda', default='0')
        self.parser.add_argument('--seed', type=int, default=1337, help='random seed')

        self.parser.add_argument('--model', type=str, default='pix2pixHD')
        self.parser.add_argument('--L1_Loss', type=bool, default=True)
        self.parser.add_argument('--SSIM_Loss', type=bool, default=True)


        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512)
        self.parser.add_argument('--tf_log', action='store_true')
        self.parser.add_argument('--name', type=str, default='OCT_Denoise')

        self.parser.add_argument('--norm', type=str, default='batch')


        self.parser.add_argument('--no_html', action='store_true')
        self.parser.add_argument('--num_D', type=int, default=2)
        self.parser.add_argument('--n_layers_D', type=int, default=3)
        self.parser.add_argument('--ndf', type=int, default=64)
        self.parser.add_argument('--no_ganFeat_loss', action='store_true')
        self.parser.add_argument('--no_lsgan', action='store_true')
        self.parser.add_argument('--pool_size', type=int, default=0)



        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        return self.opt
