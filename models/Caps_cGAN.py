import torch
from models.base_model import BaseModel
from models import networks
from models.SSIM import SSIM


class SSIM_Loss(torch.nn.Module):
    def __init__(self):
        super(SSIM_Loss, self).__init__()
        self.criterion = SSIM()

    def forward(self, x_generate, y_label):
        loss = 0
        for i in range(x_generate.size(0)):
            loss += self.criterion(x_generate[i].unsqueeze(0), y_label[i].unsqueeze(0))
        return loss / float(x_generate.size(0))


class Caps_cGAN(BaseModel):
    def name(self):
        return 'Caps_cGAN'

    def init_loss_filter(self, use_L1_loss, use_SSIM_loss):
        flags = (True, use_L1_loss, use_SSIM_loss, True, True)

        def loss_filter(g_gan, g_l1_feat, g_ssim, d_real, d_fake):
            return [l for (l, f) in zip((g_gan, g_l1_feat, g_ssim, d_real, d_fake), flags) if f]

        return loss_filter

    def save(self, which_epoch, T_S):
        self.save_network(self.netG, '{}_G'.format(T_S), which_epoch, self.gpu_ids)
        self.save_network(self.netD, '{}_D'.format(T_S), which_epoch, self.gpu_ids)

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.isTrain = opt.isTrain
        input_nc = opt.input_ch

        ##### define networks
        # Generator network
        netG_input_nc = input_nc

        self.netG = networks.define_G(netG_input_nc, opt.out_ch, opt.netG,gpu_ids=self.gpu_ids, opt=opt)

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.out_ch

            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)


        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(opt.L1_Loss, opt.SSIM_Loss)

            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionL1_Loss = torch.nn.L1Loss()
            self.criterionSSIM_Loss = SSIM_Loss()

            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN', 'L1_Loss', 'SSIM_Loss', 'D_real', 'D_fake')

            params = list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def discriminate(self, input_label, test_image):
        # print("input_label == {},test_image == {}".format(input_label.shape, test_image.shape))
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        return self.netD.forward(input_concat)

    def forward(self,image,label,Super=True,infer=False):
        # print("labeled_bs =========== {}".format(labeled_bs))

        # Encode Inputs
        image, label = image.cuda(),label.cuda()
        fake_image = self.netG.forward(image)

        # Fake Detection and Loss
        pred_fake_pool_Super_input = self.discriminate(image, fake_image)
        loss_D_fake_input = self.criterionGAN(pred_fake_pool_Super_input, False)
        pred_fake_pool_Super_label = self.discriminate(label, fake_image)
        loss_D_fake_label = self.criterionGAN(pred_fake_pool_Super_label, False)

        loss_D_fake = loss_D_fake_input+loss_D_fake_label


        # GAN loss (Fake Passability Loss)
        pred_fake_Super_input = self.netD.forward(torch.cat((image, fake_image), dim=1))
        loss_G_GAN_input = self.criterionGAN(pred_fake_Super_input, True)

        pred_fake_Super_label = self.netD.forward(torch.cat((label, fake_image), dim=1))
        loss_G_GAN_label = self.criterionGAN(pred_fake_Super_label, True)

        loss_G_GAN = loss_G_GAN_input+loss_G_GAN_label

        if Super:
            # Real Detection and Loss
            pred_real_input = self.discriminate(image, label)
            loss_D_real_input = self.criterionGAN(pred_real_input, True)

            pred_real_label = self.discriminate(label, label)
            loss_D_real_label = self.criterionGAN(pred_real_label, True)
            loss_D_real = loss_D_real_input+loss_D_real_label


            loss_L1 = 100.0 * self.criterionL1_Loss(fake_image, label)
            loss_SSIM = 10.0 * self.criterionSSIM_Loss(fake_image, label)
            # Only return the fake_B image if necessary to save BW
            return self.loss_filter(loss_G_GAN, loss_L1, loss_SSIM, loss_D_real, loss_D_fake),fake_image
        else:
            # Only return the fake_B image if necessary to save BW
            return self.loss_filter(loss_G_GAN, False, False, False, loss_D_fake),fake_image


class InferenceModel(Caps_cGAN):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)
