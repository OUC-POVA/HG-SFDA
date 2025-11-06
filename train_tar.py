import os, sys

sys.path.append("./")
from tqdm import tqdm
import pickle
from utils import *
from torch import autograd
import argparse
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms import Normalize
import network
# import loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import torchvision.transforms as transforms
import warnings

warnings.filterwarnings('ignore')
import time
from sklearn.metrics import accuracy_score
from hyperg.generation.sparse import gen_l1_hg
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from sklearn.manifold import TSNE
from sklearn.manifold import spectral_embedding
from sklearn.manifold import SpectralEmbedding
from scipy.sparse.csgraph import laplacian
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

mp.set_sharing_strategy('file_system')
from torch.utils.data import WeightedRandomSampler
from hyperg import hgnn
import psutil

from fvcore.nn import FlopCountAnalysis, parameter_count
def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log2(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    c = torch.log2(torch.tensor(args.class_num))
    
    w = entropy / c
    w = torch.exp(w)
     
    return w



def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=1):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr0"] * decay
        param_group["weight_decay"] = 1e-3
        param_group["momentum"] = 0.9
        param_group["nesterov"] = True
    return optimizer


class ELR_loss(nn.Module):
    def __init__(self, beta, lamb, gamma, num, cls):
        super(ELR_loss, self).__init__()
        self.ema = torch.zeros(num, cls).cuda()
        self.beta = beta
        self.lamb = lamb
        self.gamma = gamma

    def forward(self, index, outputs):
        y_pred = torch.nn.functional.softmax(outputs, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()

        self.ema[index] = self.beta * self.ema[index] + (1 - self.beta) * (
                    (y_pred_) / (y_pred_).sum(dim=1, keepdim=True))
        elr_reg = ((1 - (self.ema[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = self.lamb * elr_reg
        return final_loss


class ImageList_idx(Dataset):
    def __init__(
            self, image_list, labels=None, transform=None, target_transform=None, mode="RGB"
    ):
        imgs = make_dataset(image_list, labels)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == "RGB":
            self.loader = rgb_loader
        elif mode == "L":
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        # for visda
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)


def office_load_idx(args):
    train_bs = args.batch_size
    if args.home == True:
        ss = args.dset.split("2")[0]
        tt = args.dset.split("2")[1]
        if ss == "a":
            s = "Art"
        elif ss == "c":
            s = "Clipart"
        elif ss == "p":
            s = "Product"
        elif ss == "r":
            s = "Real_World"

        if tt == "a":
            t = "Art"
        elif tt == "c":
            t = "Clipart"
        elif tt == "p":
            t = "Product"
        elif tt == "r":
            t = "Real_World"

        s_tr, s_ts = "./data/office-home/{}.txt".format(
            s
        ), "./data/office-home/{}.txt".format(s)

        txt_src = open(s_tr).readlines()
        dsize = len(txt_src)

        s_tr = txt_src
        s_ts = txt_src

        t_tr, t_ts = "./data/office-home/{}.txt".format(
            t
        ), "./data/office-home/{}.txt".format(t)
        prep_dict = {}
        prep_dict["source"] = image_train()
        prep_dict["target"] = image_target()
        prep_dict["test"] = image_test()
        train_source = ImageList_idx(s_tr, transform=prep_dict["source"])
        test_source = ImageList_idx(s_ts, transform=prep_dict["source"])
        train_target = ImageList_idx(
            open(t_tr).readlines(), transform=prep_dict["target"]
        )
        txt_tar = open(t_tr).readlines()
        lab = []
        for line in txt_tar:
            lab.append(int(line.split(' ')[1]))

        train_target.targets = lab

        class_count = [0] * args.class_num
        for target in train_target.targets:
            class_count[target] += 1

        class_weights = 1.0 / torch.tensor(class_count, dtype=torch.float)
        sample_weights = [class_weights[target] for target in train_target.targets]

        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        test_target = ImageList_idx(open(t_ts).readlines(), transform=prep_dict["test"])

    dset_loaders = {}
    dset_loaders["source_tr"] = DataLoader(
        train_source,
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dset_loaders["source_te"] = DataLoader(
        test_source,
        batch_size=train_bs * 2,  # 2
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dset_loaders["target"] = DataLoader(
        train_target,
        batch_size=train_bs,
        sampler=sampler,
        num_workers=args.worker,
        drop_last=False,
    )
    dset_loaders["test"] = DataLoader(
        test_target,
        batch_size=train_bs * 3,  # 3
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    return dset_loaders


def hyper_decay(x, beta=-5, alpha=1):
    weight = (1 + 1 * x) ** (-beta) * alpha
    return weight

def train_hgat(model,optimizer,features,adj,labels):
    model.train()
    optimizer.zero_grad()
    
    g_ft,output,attention = model(features, adj)
    
    labels = labels.long().cuda()  
    output = output.cuda()
  
    ce_criterion = nn.CrossEntropyLoss().cuda()
    loss_ce = ce_criterion(output, labels) 
    
    mse_criterion = nn.MSELoss().cuda()
    loss_mr = mse_criterion(g_ft.cuda(), features.cuda())
    
    loss_train = 0.4 * loss_ce + 0.6 * loss_mr
    
    loss_train.backward()
    optimizer.step()
  
    #print("loss_train",loss_train)

    return loss_train.data.item()



def train_target_decay(args):
    dset_loaders = office_load_idx(args)
    ## set base network

    netF = network.ResNet_FE().cuda()
    oldC = network.feat_classifier(
        type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck
    ).cuda()

    modelpath = args.output_dir + "/source_F.pt"
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir + "/source_C.pt"
    oldC.load_state_dict(torch.load(modelpath))

    optimizer = optim.SGD(
        [
            {"params": netF.feature_layers.parameters(), "lr": args.lr * 0.1},  # .1
            {"params": netF.bottle.parameters(), "lr": args.lr * 1},  # 1
            {"params": netF.bn.parameters(), "lr": args.lr * 1},  # 1
            {"params": oldC.parameters(), "lr": args.lr * 1},  # 1
        ],
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )
    optimizer = op_copy(optimizer)
    #HGNN
    Embedding = hgnn.HGNN_GAT(args.class_num, 256)
    optimizer_c = optim.SGD(Embedding.parameters(), 0.001, momentum=0.9) 
    mse_criterion = nn.MSELoss().cuda()
    ce_criterion = nn.CrossEntropyLoss().cuda()

    acc_init = 0
    start = True
    loader = dset_loaders["target"]
    num_sample = len(loader.dataset)
    print("dataset_size", num_sample)

    ##
    fea_bank = torch.randn(num_sample, 256)
    score_bank = torch.randn(num_sample, args.class_num).cuda()
    hoop_bank = torch.randn(num_sample).cuda()
    ema = torch.zeros(num_sample, args.class_num).cuda()
    netF.eval()
    oldC.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.__next__()
            inputs = data[0]
            indx = data[-1]
            # labels = data[1]
            inputs = inputs.cuda()                    
            #print("inputs",inputs.size())
            #####
            output = netF.forward(inputs)  # a^t
            output_norm = F.normalize(output)
            outputs = oldC(output)
            y_pred = torch.nn.functional.softmax(outputs, dim=1)
            y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
            y_pred_ = y_pred.data.detach()
            ema[indx] = args.beta1 * ema[indx] + (1 - args.beta1) * ((y_pred_) / (y_pred_).sum(dim=1, keepdim=True))
            outputs = nn.Softmax(-1)(outputs)
            if args.sharp:
                outputs = outputs ** 2 / ((outputs ** 2).sum(dim=0))
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  # .cpu()

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    hyper_interval = math.ceil(max_iter / args.hyper_iter)#max_iter // args.hyper_iter
    print("hyper_interval",hyper_interval)
    epoch_10 = 10 * len(dset_loaders["target"])
    iter_num = 0
    j = 0
    loss_list = []
    acc_list = []

    netF.train()
    oldC.train()
    p = 0
    start_time = time.time()
    while iter_num < max_iter:

        netF.train()
        oldC.train()

        try:
            inputs_test, _, tar_idx = iter_target.__next__()
        except:
            iter_target = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_target.__next__()

        if inputs_test.size(0) == 1:
            continue

        if args.alpha_decay:
            alpha = (1 + 10 * iter_num / max_iter) ** (-args.beta) * args.alpha
        else:
            alpha = args.alpha

        inputs_test = inputs_test.cuda()

        iter_num += 1
        if args.lr_decay:
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_target = inputs_test.cuda()

        features_test = netF(inputs_target)

        output = oldC(features_test)
        softmax_out = nn.Softmax(dim=1)(output)
        output_f_norm1 = F.normalize(features_test)
        output_re = softmax_out.unsqueeze(1)  # batch x 1 x num_class
        y_pred = torch.nn.functional.softmax(output, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()
        ema[tar_idx] = args.beta1 * ema[tar_idx] + (1 - args.beta1) * ((y_pred_) / (y_pred_).sum(dim=1, keepdim=True))
        with torch.no_grad():
            output_f_norm = F.normalize(features_test)
            output_f_ = output_f_norm.cpu().detach().clone()

            pred_bs = softmax_out

            fea_bank[tar_idx] = output_f_.detach().clone().cpu()
            score_bank[tar_idx] = pred_bs.detach().clone()

        if iter_num == 1 or iter_num % hyper_interval == 0:
          
                # Construct hypergraph
                hg = gen_l1_hg(fea_bank,hoop_bank,score_bank,gamma=1, n_neighbors=args.hyper_edge,n_clusters=args.class_num)
                
                adj = hg.incident_matrix()
                adj1 = adj.toarray()
                
                for _ in range(20):
                  _, preds = torch.max(score_bank, 1)  
                  train_hgat(Embedding,optimizer_c,fea_bank, adj1,preds) #mem_label
                  
                adj_k = adj.copy()
                for i in range(args.hyper_edge - 1):
                    adj_k = adj_k.dot(adj)
             
                tsne = TSNE(n_components=2)
                X_embedded = tsne.fit_transform(adj_k.toarray())#.toarray()

                nbrs = NearestNeighbors(n_neighbors=args.K + 1, metric='euclidean')
                nbrs.fit(X_embedded)
                indices = nbrs.kneighbors(return_distance=False)
                indices = indices[:, 1:]  # num_sample x K
                j = j + 1
                print("Optimisation of hypergraph:", j, "times")
       

        idx_near = indices[tar_idx]
        score_near = score_bank[idx_near, :]  # batch x K x C
        
        softmax_out_un = softmax_out.unsqueeze(1).expand(
            -1, args.K, -1
        ).cuda()  # batch x K x C
        epsilon = 1e-10
        softmax_out_un = torch.clamp(softmax_out_un, epsilon, 1.0 - epsilon)
        score_near = torch.clamp(score_near, epsilon, 1.0 - epsilon)

        distance = torch.norm(softmax_out_un - score_near, dim=2)


        distance1 = (distance - torch.min(distance)) / (torch.max(distance) - torch.min(distance))
        
        loss = torch.mean(
            ((1 - distance1 ** args.gamma) * (F.kl_div(softmax_out_un, score_near.cuda(), reduction="none").sum(-1))).sum(1))

        # other prediction scores as negative pairs
        mask = torch.ones((inputs_target.shape[0], inputs_target.shape[0]))
        diag_num = torch.diag(mask)
        mask_diag = torch.diag_embed(diag_num)
        mask = mask - mask_diag
        if args.noGRAD:
            copy = softmax_out.T.detach().clone()
        else:
            copy = softmax_out.T  # .detach().clone()

        dot_neg = softmax_out @ copy  # batch x batch
        distance2 = torch.cdist(softmax_out, softmax_out, p=2)

        distance = (distance2 - torch.min(distance2)) / (torch.max(distance2) - torch.min(distance2))
        distance2 = distance * mask.cuda()
        
        dot_neg = ((1 - distance2 ** args.gamma) * dot_neg * mask.cuda()).sum(-1)  # batch
       
        neg_pred = torch.mean(dot_neg)
        loss += neg_pred * alpha

        criterion_elr = torch.mean(
            F.kl_div(softmax_out, ema[tar_idx], reduction="none").sum(1)
        )
        loss += args.lamb * criterion_elr
     
        loss_cpu = loss.cpu().detach().clone()
        loss_list.append(loss_cpu)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        end_time = time.time()
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            oldC.eval()

            acc1, _ = cal_acc_(dset_loaders["test"], netF, oldC)  # 1

            log_str = "Task: {}, Iter:{}/{}; Accuracy on target = {:.2f}%".format(
                args.dset, iter_num, max_iter, acc1 * 100
            )
            args.out_file.write(log_str + "\n")
            args.out_file.flush()
            print(log_str + "\n")
            total_time = end_time - start_time
            print("training time:", total_time / 60, "minutes")

            acc_list.append(acc1 * 100)

            
            if acc1 >= acc_init:
                acc_init = acc1
                best_netF = netF.state_dict()
                best_netC = oldC.state_dict()

                torch.save(
                    best_netF,
                    osp.join(
                        args.output_dir, "Fbest_{}_{}.pt".format(args.file, args.seed)
                    ),
                )
                torch.save(
                    best_netC,
                    osp.join(args.output_dir, "Cbest_{}_{}.pt").format(
                        args.file, args.seed
                    ),
                )
             
             #####
        """
        if iter_num % epoch_10 == 0:
            p = p + 1
            print("epoch_{}",p)
            best_netF = netF.state_dict()
            best_netC = oldC.state_dict()

            torch.save(
                    best_netF,
                    osp.join(
                        args.output_dir, "Fepoch_{}_{}_{}.pt".format(p,args.file, args.seed)
                    ),)
            torch.save(
                    best_netC,
                    osp.join(args.output_dir, "Cepoch_{}_{}_{}.pt").format(p,args.file, args.seed),)

    #
         """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Domain Adaptation on office-home dataset"
    )
    parser.add_argument(
        "--gpu_id", type=str, nargs="?", default="1", help="device id to run"
    )
    parser.add_argument("--s", type=int, default=0, help="source")
    parser.add_argument("--t", type=int, default=1, help="target")
    parser.add_argument("--max_epoch", type=int, default=40, help="maximum epoch")
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--worker", type=int, default=8, help="number of workers")
    parser.add_argument("--k", type=int, default=2, help="number of neighborhoods")
    parser.add_argument("--dset", type=str, default="a2r")
    parser.add_argument("--choice", type=str, default="shot")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--seed", type=int, default=2021, help="random seed")
    parser.add_argument("--class_num", type=int, default=65)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--hyper_edge", type=int, default=5)
    parser.add_argument("--hyper_iter", type=int, default=13)
    parser.add_argument("--par", type=float, default=0.1)
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--layer", type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument("--classifier", type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument("--smooth", type=float, default=0.1)
    parser.add_argument("--output", type=str, default="office-home_weight")  
    parser.add_argument("--file", type=str, default="noDIV")
    parser.add_argument("--home", action="store_true")
    parser.add_argument("--NOneg", default=False, action="store_true")
    parser.add_argument("--affi_neg", default=False, action="store_true")
    parser.add_argument("--ori", default=False, action="store_true")
    parser.add_argument("--no2hop", default=False, action="store_true")
    parser.add_argument("--onlyNN", default=False, action="store_true")
    parser.add_argument("--self", default=False, action="store_true")
    parser.add_argument("--cc", default=False, action="store_true")
    parser.add_argument("--alpha_decay", default=True, action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--topKNEG", default=False, action="store_true")
    parser.add_argument("--conf", default=False, action="store_true")
    parser.add_argument("--lr_decay", default=False, action="store_true")
    parser.add_argument("--r_batch", default=False, action="store_true")
    parser.add_argument("--noGRAD", default=False, action="store_true")
    parser.add_argument("--sharp", default=False, action="store_true")
    parser.add_argument("--sharp_neg", default=False, action="store_true")
    parser.add_argument('--beta1', default=0.5, type=float, help='ema for t')
    parser.add_argument('--lamb', default=2.0, type=float, help='elr loss hyper parameter')
    parser.add_argument('--gamma', default=7.0, type=float)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = 2022
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    current_folder = "./"
    args.output_dir = osp.join(
        current_folder, args.output, "seed" + str(2021), args.dset
    )
    if not osp.exists(args.output_dir):
        os.system("mkdir -p " + args.output_dir)
    args.out_file = open(osp.join(args.output_dir, args.file + ".txt"), "w")
    args.out_file.write(print_args(args) + "\n")
    args.out_file.flush()

    train_target_decay(args)

