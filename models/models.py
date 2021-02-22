import torch
def create_model(opt):
    from .Caps_cGAN import Caps_cGAN, InferenceModel
    if opt.isTrain:
        model = Caps_cGAN()
    else:
        model = InferenceModel()
    model.initialize(opt)
    # if opt.verbose:
    #     print("model [%s] was created" % (model.name()))
    if opt.isTrain and len(opt.gpu_ids):# and not opt.fp16:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
