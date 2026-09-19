import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import time
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import roc_auc_score


def KL(alpha):

    ones = torch.ones([1, alpha.shape[-1]], dtype=torch.float32, device=alpha.device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (alpha - ones).mul(torch.digamma(alpha) - torch.digamma(sum_alpha)).sum(dim=1, keepdim=True)
    kl = first_term + second_term
    return kl.reshape(-1)


def loss_log(alpha, labels, kl_penalty):
    y = F.one_hot(labels.long(), alpha.shape[-1])
    log_likelihood = torch.sum(y * (torch.log(alpha.sum(dim=-1, keepdim=True)) - torch.log(alpha)), dim=-1)

    loss = log_likelihood + kl_penalty * KL((alpha - 1) * (1 - y) + 1)
    return loss


def loss_digamma(alpha, labels, kl_penalty):
    y = F.one_hot(labels.long(), alpha.shape[-1])
    log_likelihood = torch.sum(y * (torch.digamma(alpha.sum(dim=-1, keepdim=True)) - torch.digamma(alpha)), dim=-1)
    loss = log_likelihood + kl_penalty * KL((alpha - 1) * (1 - y) + 1)
    return loss


def loss_mse(alpha, labels, kl_penalty):
    y = F.one_hot(labels.long(), alpha.shape[-1])
    sum_alpha = torch.sum(alpha, dim=-1, keepdim=True)
    err = (y - alpha / sum_alpha) ** 2
    var = alpha * (sum_alpha - alpha) / (sum_alpha ** 2 * (sum_alpha + 1))
    loss = torch.sum(err + var, dim=-1)
    loss = loss + kl_penalty * KL((alpha - 1) * (1 - y) + 1)
    return loss


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return torch.reshape(x, self.shape)


class InferNet(nn.Module):
    def __init__(self, layers_dim, dropout=0.2):
        super().__init__()
        self.fc = nn.Sequential()
        for i in range(len(layers_dim) - 1):
            self.fc.add_module(f'infer{i}', nn.Linear(layers_dim[i], layers_dim[i + 1]))
            self.fc.add_module(f'dropout{i}', nn.Dropout(dropout))
            self.fc.add_module(f'relu{i}', nn.ReLU())

    def forward(self, x):
        return self.fc(x)


class EML(nn.Module):
    def __init__(self, sample_shapes: list, num_classes: int, device, delta=0.1 ):
        super().__init__()
        self.device = device
        self.num_views = len(sample_shapes)
        self.num_classes = num_classes
        self.delta = delta
        self.modeloss = 'EML'
        self.ModalPre = None
        self.hsic_factor = 0.5
        
        self.c = nn.ModuleList([
            nn.Sequential(
                Reshape([-1, 1, 23, 256]),
                nn.Conv2d(1, 1, (1, 128)),
                nn.BatchNorm2d(1),
                nn.Tanh(),
                nn.Conv2d(1, 30, (1, 65)),
                nn.BatchNorm2d(30),
                nn.Tanh(),
                nn.Conv2d(30, 20, (4, 33)),
                nn.BatchNorm2d(20),
                nn.Tanh(),
                nn.Conv2d(20, 10, (8, 18)),
                nn.BatchNorm2d(10),
                nn.Tanh(),
                Reshape([-1, 13 * 16 * 10]),
                nn.Linear(13 * 16 * 10, 1024),
                nn.Tanh()
            ),
            nn.Sequential(
                Reshape([-1, 1, 23, 27]),
                nn.Conv2d(1, 20, (4, 4)),
                nn.BatchNorm2d(20),
                nn.Tanh(),
                nn.Conv2d(20, 10, (8, 8)),
                nn.BatchNorm2d(10),
                nn.Tanh(),
                Reshape([-1, 13 * 17 * 10]),
                nn.Linear(13 * 17 * 10, 1024),
                nn.Tanh()
            ),
            nn.Sequential(
                Reshape([-1, 1, 23, 14, 256]),
                nn.Conv3d(1, 1, (1, 1, 129)),
                nn.BatchNorm3d(1),
                nn.Tanh(),
                nn.Conv3d(1, 30, (4, 4, 65)),
                nn.BatchNorm3d(30),
                nn.Tanh(),
                nn.Conv3d(30, 20, (4, 4, 33)),
                nn.BatchNorm3d(20),
                nn.Tanh(),
                nn.Conv3d(20, 10, (8, 1, 17)),
                nn.BatchNorm3d(10),
                nn.Tanh(),
                Reshape([-1, 10 * 10 * 8 * 16]),
                nn.Linear(10 * 10 * 8 * 16, 2048),
                nn.Tanh(),
                nn.Linear(2048, 1024),
                nn.Tanh()
            ),
        ])  
        self.f = nn.ModuleList([InferNet([1024, 512, 128]) for i in range(self.num_views)])
        self.g = nn.ModuleList([InferNet([128, 64, num_classes]) for i in range(self.num_views)])
        self.to(device)

    def forward(self, x: dict, target=None, kl_penalty=0, ModelLoss='EML', iter_num=None):
        view_x = dict()
        for v in x.keys():
            view_x[v] = self.c[v](x[v].to(self.device))  

        view_h = dict()
        for v in view_x.keys():
            view_h[v] = self.f[v](view_x[v])

        view_e = dict()
        for v in view_h.keys():
            view_e[v] = self.g[v](view_h[v])

        fusion_e = torch.zeros_like(view_e[0])
        for v in view_e.keys():
            fusion_e = (fusion_e + view_e[v]) / 2

        loss = None
        epoch = 30
        gamma = 1

        if target is not None:
            loss = calculate_eml_loss(view_e, fusion_e, target, self.device, epoch, gamma)
            if ModelLoss == 'HSIC' and iter_num is not None:
                
                
                _, _, _, view_h_bias = self.ModelPre(x)
                
                
                if iter_num % 2 == 1:
                    loss_hsic_f = sum(self.hsic_factor * hsic_loss(view_h[v], view_h_bias[v].detach(), unbiased=True) for v in view_h_bias.keys())
                    loss += loss_hsic_f.mean()
                else:
                    loss_hsic_g = sum(-self.hsic_factor * hsic_loss(view_h[v].detach(), view_h_bias[v], unbiased=True) for v in view_h_bias.keys())
                    

                    optimizer_pre = torch.optim.Adam(self.ModelPre.parameters(), lr=1e-4)
                    optimizer_pre.zero_grad()

                    view_e_bias = dict()
                    for v in view_h_bias.keys():
                        view_e_bias[v] = self.g[v](view_h_bias[v])

                    fusion_e_bias = torch.zeros_like(view_e_bias[0])
                    for v in view_e_bias.keys():
                        fusion_e_bias = (fusion_e_bias + view_e_bias[v]) / 2

                    total_loss_pre = loss_hsic_g.mean() + calculate_eml_loss(view_e_bias, fusion_e_bias, target, self.device, epoch=20, gamma=1)
                    total_loss_pre.backward(retain_graph=True)
                    optimizer_pre.step()

        return view_e, fusion_e, loss, view_h


def calculate_eml_loss(view_e, fusion_e, target, device, epoch, gamma):
    loss_fusion = compute_fmse(fusion_e + 1, target, epoch)
    loss_views = sum(compute_fmse(view_e[v] + 1, target, epoch) for v in view_e.keys())

    num_views = len(view_e)
    if num_views > 0:
        batch_size, num_classes = fusion_e.shape[0], fusion_e.shape[1]
        p = torch.zeros((num_views, batch_size, num_classes)).to(device)
        u = torch.zeros((num_views, batch_size)).to(device)
        dc_mode = 'fusion_legacy'
        for v in range(num_views):
            if dc_mode == 'fusion_legacy':
                alpha = fusion_e[v] + 1
            elif dc_mode == 'view':
                alpha = view_e[v] + 1
            else:
                raise ValueError("dc_mode must be 'fusion_legacy' or 'view'")
            
            S = torch.sum(alpha, dim=-1, keepdim=True)
            p[v] = alpha / S
            u[v] = torch.squeeze(num_classes / S)
        dc_sum = 0
        for i in range(num_views):
            pd = torch.sum(torch.abs(p - p[i]) / 2, dim=2) / (num_views - 1)
            cc = (1 - u[i]) * (1 - u)
            dc = pd * cc
            dc_sum = dc_sum + torch.sum(dc, dim=0)
        dc_loss = torch.mean(dc_sum.to(device))
    else:
        dc_loss = torch.tensor(0.0, device=device)

    return loss_fusion.mean() + loss_views.mean() + gamma * dc_loss




def compute_fmse(evi_alp_, labels_, epoch):
    target_concentration = 100
    fisher_c = 0.01
    labels_1hot_ = torch.zeros(int(evi_alp_.size(0)), int(evi_alp_.size(1))).cuda().scatter_(1, labels_.view(-1, 1), 1)
    
    
    
    
    evi_alp0_ = torch.sum(evi_alp_, dim=-1, keepdim=True)
    
    

    gamma1_alp = torch.polygamma(1, evi_alp_)
    gamma1_alp0 = torch.polygamma(1, evi_alp0_)

    gap = labels_1hot_ - evi_alp_ / evi_alp0_

    loss_mse_ = ((gap.pow(2) * gamma1_alp).sum(-1)).mean() / 3.

    loss_var_ = (evi_alp_ * (evi_alp0_ - evi_alp_) * gamma1_alp / (evi_alp0_ * evi_alp0_ * (evi_alp0_ + 1))).sum(
        -1).mean() / 3.
    
    

    loss_det_fisher_ = - (
                torch.log(gamma1_alp).sum(-1) + torch.log(1.0 - (gamma1_alp0 / gamma1_alp).sum(-1))).mean() / 3.

    
    loss_kl_ = compute_kl_loss(evi_alp_, target_concentration, labels_) / 3.
    regr = np.minimum(1.0, (epoch + 1) / 10.)

    loss = 40 * loss_mse_ + 50 * loss_var_ + fisher_c * loss_det_fisher_ + regr * 0.05 * loss_kl_

    return loss


def compute_kl_loss(alphas,  target_concentration=1, labels=None, concentration=1.0, epsilon=1e-8):

    if target_concentration < 1.0:
        concentration = target_concentration

    target_alphas = torch.ones_like(alphas) * concentration
    if labels is not None:
        target_alphas += torch.zeros_like(alphas).scatter_(-1, labels.unsqueeze(-1), target_concentration - 1)

    alp0 = torch.sum(alphas, dim=-1, keepdim=True)
    target_alp0 = torch.sum(target_alphas, dim=-1, keepdim=True)

    alp0_term = torch.lgamma(alp0 + epsilon) - torch.lgamma(target_alp0 + epsilon)
    alp0_term = torch.where(torch.isfinite(alp0_term), alp0_term, torch.zeros_like(alp0_term))
    assert torch.all(torch.isfinite(alp0_term)).item()

    alphas_term = torch.sum(torch.lgamma(target_alphas + epsilon) - torch.lgamma(alphas + epsilon)
                            + (alphas - target_alphas) * (torch.digamma(alphas + epsilon) -
                                                            torch.digamma(alp0 + epsilon)), dim=-1, keepdim=True)
    alphas_term = torch.where(torch.isfinite(alphas_term), alphas_term, torch.zeros_like(alphas_term))
    assert torch.all(torch.isfinite(alphas_term)).item()

    loss = (torch.squeeze(alp0_term + alphas_term)).mean()

    return loss


def _kernel(X, sigma):
    X = X.view(len(X), -1)
    XX = X @ X.t()
    X_sqnorms = torch.diag(XX)
    X_L2 = -2 * XX + X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0)
    gamma = 1 / (2 * sigma ** 2)

    kernel_XX = torch.exp(-gamma * X_L2)
    return kernel_XX

def hsic_loss(input1, input2, unbiased=False):
    N = len(input1)
    if N < 4:
        return torch.tensor(0.0).to(input1.device)
    
    sigma_x = np.sqrt(input1.size()[1])
    sigma_y = np.sqrt(input2.size()[1])

    
    kernel_XX = _kernel(input1, sigma_x)
    kernel_YY = _kernel(input2, sigma_y)

    if unbiased:
        tK = kernel_XX - torch.diag(torch.diag(kernel_XX))
        tL = kernel_YY - torch.diag(torch.diag(kernel_YY))
        hsic = (
            torch.trace(tK @ tL)
            + (torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2))
            - (2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / (N - 2))
        )
        loss = hsic 
    else:
        KH = kernel_XX - kernel_XX.mean(0, keepdim=True)
        LH = kernel_YY - kernel_YY.mean(0, keepdim=True)
        loss = torch.trace(KH @ LH / (N - 1) ** 2)
    return loss


