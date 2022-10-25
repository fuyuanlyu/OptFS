import sys, os
import torch
import pickle
from data import tfloader
import modules.models as models

def getModel(model:str, opt):
    model = model.lower()
    if model == "fm":
        return models.FM(opt)
    elif model == "deepfm":
        return models.DeepFM(opt)
    elif model == "ipnn":
        return models.InnerProductNet(opt)
    elif model == "dcn":
        return models.DeepCrossNet(opt)
    else:
        raise ValueError("Invalid model type: {}".format(model))

def getOptim(network, optim, lr, l2):
    params = network.parameters()
    optim = optim.lower()
    if optim == "sgd":
        return torch.optim.SGD(params, lr= lr, weight_decay = l2)
    elif optim == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay = l2)
    else:
        raise ValueError("Invalid optmizer type:{}".format(optim))

def getDevice(device_id):
    if device_id != -1:
        assert torch.cuda.is_available(), "CUDA is not available"
        # torch.cuda.set_device(device_id)
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def getDataLoader(dataset:str, path):
    dataset = dataset.lower()
    if dataset == 'criteo':
        return tfloader.CriteoLoader(path)
    elif dataset == 'avazu':
        return tfloader.Avazuloader(path)
    elif dataset == 'kdd12':
        return tfloader.KDD12loader(path)

def get_stats(path):
    defaults_path = os.path.join(path + "/defaults.pkl")
    with open(defaults_path, 'rb') as fi:
        defaults = pickle.load(fi)
    offset_path = os.path.join(path + "/offset.pkl")
    with open(offset_path, 'rb') as fi:
        offset = pickle.load(fi)
    # return [i+1 for i in list(defaults.values())] 
    return list(defaults.values()), list(offset.values())
