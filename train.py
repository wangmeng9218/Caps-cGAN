from options.train_options import TrainOptions
from models.models import create_model
import torch
import os
from dataprepare.slit_loader import Slit_loader
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import cv2
import util.util as u
from dataprepare.slit_loader import TwoStreamBatchSampler
from collections import OrderedDict
from util.visualizer import Visualizer
import util.util as util


def val_Cycle(args, modelG_A, dataloader_val,epoch,k_fold):
    print('\n')
    print('Start Validation......')
    with torch.no_grad():  # 适用于推断阶段，不需要反向传播
        modelG_A.eval()  # 测试模式，自动把BN和DropOut固定住，用训练好的值
        tbar = tqdm.tqdm(dataloader_val, desc='\r')  # 添加一个进度提示信息，只需要封装任意的迭代器 tqdm(iterator)，desc进度条前缀


        if not os.path.exists(os.path.join(args.checkpoints_dir,args.net_work,"Gen_B/{}/{}".format(k_fold,epoch))):
            os.makedirs(os.path.join(args.checkpoints_dir,args.net_work,"Gen_B/{}/{}".format(k_fold,epoch)))

        for i, (data, _, _) in enumerate(tbar):  # i=300/batch_size=3
            # tbar.update()
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
            predict = modelG_A.forward(data,isTrain=False)
            predict = (predict+1)/2

            data_num = data.size(0)
            for idx in range(data_num):
                gen_B = np.array(predict[idx].squeeze(0).cpu().numpy()*255,dtype=np.uint8)
                print(os.path.join(args.checkpoints_dir,args.net_work,"Gen_B/{}/{}/{}_{}_{}.png".format(k_fold,epoch,epoch,i,idx)))
                cv2.imwrite(
                    os.path.join(args.checkpoints_dir,args.net_work,"Gen_B/{}/{}/{}_{}_{}.png".format(k_fold,epoch,epoch,i,idx)),
                    gen_B)



def train(opt, model, dataloader_train_Super,dataloader_train ,dataloader_val, optimizer_G,optimizer_D, k_fold):
    total_steps = 0.0
    for epoch in range(opt.num_epochs):
        lr_G = u.adjust_learning_rate(opt,optimizer_G,epoch)    # update learning rates in the beginning of every epoch.
        lr_D = u.adjust_learning_rate(opt,optimizer_D,epoch)    # update learning rates in the beginning of every epoch.


        if epoch < 5:

            for i_Super, (data, label,img_list) in enumerate(dataloader_train_Super): # inner loop within one epoch
                model.module.netG.train()
                model.module.netD.train()
                total_steps += opt.batch_size
                if torch.cuda.is_available() and opt.use_gpu:
                    data = data.cuda()
                    label = label.cuda()  # 索引值格式为long
                losses_Pre, generated = model(data,label)   # calculate loss functions, get gradients, update network weights
                losses_Pre = [torch.mean(x) if not isinstance(x, int) else x for x in losses_Pre]
                loss_dict_Pre = dict(zip(model.module.loss_names, losses_Pre))

                # calculate final loss scalar
                loss_D = (loss_dict_Pre['D_fake'] + loss_dict_Pre['D_real']) * 0.5
                loss_G = loss_dict_Pre['G_GAN'] + loss_dict_Pre.get('L1_Loss', 0) + loss_dict_Pre.get('SSIM_Loss', 0)
                errors_Pre = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict_Pre.items()}
                ############### Backward Pass ####################
                model.module.netD.eval()
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

                model.module.netD.train()
                model.module.netG.eval()
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()



                save_fake = total_steps % 100 == 0
                if save_fake:
                    visualizer.plot_current_errors(errors_Pre, total_steps)
                    errors_Pre = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict_Pre.items()}
                    print("Pre Epoch == {}, errors ======== {}".format(epoch,errors_Pre))
                    visuals = OrderedDict([('Pre_Input_Super', util.tensor2im(data[0])),
                                           ('Pre_Denoised_Super', util.tensor2im(generated.data[0])),
                                           ('Pre_Label_Super', util.tensor2im(label[0]))])
                    visualizer.display_current_results(visuals, epoch, total_steps)

                    ### save latest model
                    print('saving the latest Pretrained model (epoch %d, total_steps %d)' % (epoch, total_steps))
                    model.module.save('Pre_trained', T_S="Pre")


            val_Cycle(args=opt, modelG_A=model.module.netG, dataloader_val=dataloader_val, epoch=epoch, k_fold=k_fold)
            print('saving the Pretrained model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.module.save('Pre_trained', T_S="Pre")
            model.module.save(epoch, T_S="Pre")

        else:

            for i, (data, label,img_list_Semi) in enumerate(dataloader_train): # inner loop within one epoch
                model.module.netG.train()
                model.module.netD.train()
                total_steps += opt.batch_size
                if torch.cuda.is_available() and opt.use_gpu:
                    data = data.cuda()
                    label = label.cuda()  # 索引值格式为long

                data_Super = data[:opt.labeled_bs]
                label_Super = label[:opt.labeled_bs]
                data_Un = data[opt.labeled_bs:]
                label_Un = label[opt.labeled_bs:]
                losses_Super, generated_Super = model(data_Super,label_Super,Super=True)   # calculate loss functions, get gradients, update network weights
                # sum per device losses
                losses_Super = [torch.mean(x) if not isinstance(x, int) else x for x in losses_Super]
                loss_dict_Super = dict(zip(model.module.loss_names, losses_Super))

                # calculate final loss scalar
                loss_D_Super = (loss_dict_Super['D_fake'] + loss_dict_Super['D_real']) * 0.5
                loss_G_Super = loss_dict_Super['G_GAN'] + loss_dict_Super.get('L1_Loss', 0) + loss_dict_Super.get('SSIM_Loss', 0)


                ############### Backward Pass ####################
                # update generator weights
                model.module.netD.eval()
                optimizer_G.zero_grad()
                loss_G_Super.backward()
                optimizer_G.step()

                model.module.netD.train()
                model.module.netG.eval()
                optimizer_D.zero_grad()
                loss_D_Super.backward()
                optimizer_D.step()



                model.module.netG.train()
                model.module.netD.train()
                losses_Un , generated_Un = model(data_Un,label_Un,Super=False)   # calculate loss functions, get gradients, update network weights

                # sum per device losses
                losses_Un = [torch.mean(x) if not isinstance(x, int) else x for x in losses_Un]
                loss_dict_Un = dict(zip(model.module.loss_names, losses_Un))

                # calculate final loss scalar
                loss_G_Un = loss_dict_Un['G_GAN']
                loss_D_Un = loss_dict_Un['D_fake']



                ############### Backward Pass ####################
                # update generator weights
                model.module.netG.train()
                model.module.netD.eval()
                optimizer_G.zero_grad()
                loss_G_Un.backward()
                optimizer_G.step()

                model.module.netG.eval()
                model.module.netD.train()
                optimizer_D.zero_grad()
                loss_D_Un.backward()
                optimizer_D.step()

                save_fake = total_steps % 100 == 0
                if save_fake:
                    errors_Super = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict_Super.items()}
                    errors_un = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict_Un.items()}
                    print("Semi_img_list ========= {}".format(img_list_Semi))
                    print("Epoch == {}, errors_Super ======== {}".format(epoch,errors_Super))
                    print("Epoch == {},errors_un ======== {}".format(epoch,errors_un))
                    visuals = OrderedDict([('Smi_Input_Super', util.tensor2im(data_Super[0])),
                                           ('Smi_Denoised_Super', util.tensor2im(generated_Super.data[0])),
                                           ('Smi_Label_Super', util.tensor2im(label_Super[0])),
                                           ('Smi_Input_Un', util.tensor2im(data_Un[0])),
                                           ('Smi_Denoised_Un', util.tensor2im(generated_Un.data[0])),
                                           ])
                    visualizer.display_current_results(visuals, epoch, total_steps)
                    ### save latest model
                    print('saving the latest Semi model (epoch %d, total_steps %d)' % (epoch, total_steps))
                    model.module.save('Latest_Semi', T_S="Semi")

            for i_Super, (data, label,img_list) in enumerate(dataloader_train_Super): # inner loop within one epoch
                model.module.netG.train()
                model.module.netD.train()
                total_steps += opt.batch_size
                if torch.cuda.is_available() and opt.use_gpu:
                    data = data.cuda()
                    label = label.cuda()  # 索引值格式为long
                losses_Pre, generated = model(data,label)   # calculate loss functions, get gradients, update network weights
                losses_Pre = [torch.mean(x) if not isinstance(x, int) else x for x in losses_Pre]
                loss_dict_Pre = dict(zip(model.module.loss_names, losses_Pre))

                # calculate final loss scalar
                loss_D = (loss_dict_Pre['D_fake'] + loss_dict_Pre['D_real']) * 0.5
                loss_G = loss_dict_Pre['G_GAN'] + loss_dict_Pre.get('L1_Loss', 0) + loss_dict_Pre.get('SSIM_Loss', 0)
                errors_Pre = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict_Pre.items()}
                ############### Backward Pass ####################
                model.module.netD.eval()
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

                model.module.netD.train()
                model.module.netG.eval()
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()



                save_fake = total_steps % 100 == 0
                if save_fake:
                    visualizer.plot_current_errors(errors_Pre, total_steps)
                    errors_Pre = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict_Pre.items()}
                    print("Fine Epoch == {}, errors ======== {}".format(epoch,errors_Pre))
                    visuals = OrderedDict([('Fine_Input_Super', util.tensor2im(data[0])),
                                           ('Fine_Denoised_Super', util.tensor2im(generated.data[0])),
                                           ('Finee_Label_Super', util.tensor2im(label[0]))])
                    visualizer.display_current_results(visuals, epoch, total_steps)

                    ### save latest model
                    print('saving the latest Fine model (epoch %d, total_steps %d)' % (epoch, total_steps))
                    model.module.save('Fine_trained', T_S="Fine")

            val_Cycle(args=opt, modelG_A=model.module.netG, dataloader_val=dataloader_val, epoch=epoch, k_fold=k_fold)
            print('saving the Fine model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.module.save('Fine_trained', T_S="Fine")
            model.module.save(epoch, T_S="Fine")


def patients_to_slices(k_fold=1,total_folds=['f1','f2','f3']):
    total_folds.remove("f{}".format(k_fold))
    label_num = 0
    ref_dict = {"f1": 512, "f2": 512,
                    "f3": 768,"f4": 640}
    for fold in total_folds:
        label_num += ref_dict[fold]

    return label_num





def main(opt,k_fold):
    ################创建数据集
    dataset_path = os.path.join(opt.data, opt.dataset)
    dataset_train = Slit_loader(dataset_path, scale=(opt.crop_height, opt.crop_width), k_fold_test=k_fold,
                                mode='train')

    total_slices = len(dataset_train)
    labeled_slice = patients_to_slices(k_fold, total_folds=['f1','f2','f3'])
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, opt.batch_size, opt.batch_size-opt.labeled_bs)

    dataloader_train = DataLoader(
        dataset_train,  # 加载的数据集(Dataset对象),dataloader是一个可迭代的对象，可以像使用迭代器一样使用
        batch_sampler=batch_sampler,
        num_workers=opt.num_threads,  # 使用多进程加载的进程数，0代表不使用多进程
        pin_memory=True,  # 是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些
    )


    dataset_val = Slit_loader(dataset_path, scale=(opt.crop_height, opt.crop_width), k_fold_test=k_fold, mode='val')
    dataloader_val = DataLoader(
        dataset_val,
        # this has to be 1
        batch_size=opt.labeled_bs,  # 只选择1块gpu，batchsize=1
        shuffle=False,
        num_workers=opt.num_threads,
        pin_memory=True,
        drop_last=False
    )




    dataset_path_Super = os.path.join(opt.data, opt.dataset_Super)
    dataset_train_Super = Slit_loader(dataset_path_Super, scale=(opt.crop_height, opt.crop_width), k_fold_test=k_fold,
                                mode='train')

    print("Labeled slices is: {}".format(
        len(dataset_train_Super)))

    dataloader_train_Super = DataLoader(
        dataset_train_Super,
        batch_size=opt.labeled_bs,
        shuffle=True,
        num_workers=opt.num_threads,  # 使用多进程加载的进程数，0代表不使用多进程
        pin_memory=True,  # 是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些
        drop_last=True  # dataset中的数据个数可能不是batch_size的整数倍，drop_last为True会将多出来不足一个batch的数据丢弃
    )

    print("Labeled slices Step: {}".format(
        len(dataloader_train_Super)))


    model = create_model(opt)
    optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D
    train(opt, model, dataloader_train_Super, dataloader_train, dataloader_val,  optimizer_G,optimizer_D, k_fold=k_fold)



if __name__ == '__main__':
    from datetime import datetime
    import socket
    from tensorboardX import SummaryWriter


    opt = TrainOptions().parse()   # get training options
    visualizer = Visualizer(opt)
    seed = opt.seed  # 在训练开始时，参数的初始化是随机的，为了让每次的结果一致，我们需要设置随机种子
    u.setup_seed(seed)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')  # 程序开始运行的时间
    log_dir = os.path.join(opt.log_dirs,
                           opt.net_work + '_' + '_' + current_time + '_' + socket.gethostname())  # 获取当前计算节点名称cu01
    if not os.path.exists(os.path.dirname(log_dir)):  # 创建目录文件夹
        os.makedirs(os.path.dirname(log_dir))
    writer = SummaryWriter(log_dir=log_dir)  # 创建writer object，log会被存入指定文件夹，writer.add_scalar保存标量值到log

    main(opt, k_fold=int(3))