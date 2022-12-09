import torch
import numpy as np

def MSE(y, y_, reduce=True):
    ax = list(range(1, len(y.shape)))

    mse = torch.mean((y-y_)**2, axis=ax)
    return torch.mean(mse) if reduce else mse

def RMSE(y, y_):
    rmse = torch.sqrt(torch.mean((y-y_)**2))
    return rmse

def Gaussian_NLL(y, mu, sigma, reduce=True):
    ax = list(range(1, len(y.shape)))

    logprob = -torch.log(sigma) - 0.5*torch.log(2*np.pi) - 0.5*((y-mu)/sigma)**2
    loss = torch.mean(-logprob, axis=ax)
    return torch.mean(loss) if reduce else loss

def Gaussian_NLL_logvar(y, mu, logvar, reduce=True):
    ax = list(range(1, len(y.shape)))

    log_liklihood = 0.5 * (
        -torch.exp(-logvar)*(mu-y)**2 - torch.log(2*torch(np.pi, dtype=logvar.dtype)) - logvar
    )
    loss = torch.mean(-log_liklihood, axis=ax)
    return torch.mean(loss) if reduce else loss

def NIG_NLL(y, gamma, v, alpha, beta, reduction='mean'):
    twoBlambda = 2*beta*(1+v)

    nll = 0.5*torch.log(np.pi/v)  \
        - alpha*torch.log(twoBlambda)  \
        + (alpha+0.5) * torch.log(v*(y-gamma)**2 + twoBlambda)  \
        + torch.lgamma(alpha)  \
        - torch.lgamma(alpha+0.5)

    # if reduce = 'mean' then mean, 'sum' then batch, 'none'
    if reduction == 'mean':
        return torch.mean(nll, dim=0)
    elif reduction == 'sum':
        return torch.sum(nll)
    else:
        return nll

    # return torch.mean(nll, dim=0) if reduce else nll

def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):
    KL = 0.5*(a1-1)/b1 * (v2*torch.square(mu2-mu1))  \
        + 0.5*v2/v1  \
        - 0.5*torch.log(torch.abs(v2)/torch.abs(v1))  \
        - 0.5 + a2*torch.log(b1/b2)  \
        - (torch.lgamma(a1) - torch.lgamma(a2))  \
        + (a1 - a2)*torch.digamma(a1)  \
        - (b1 - b2)*a1/b1
    return KL

def NIG_Reg(y, gamma, v, alpha, beta, omega=0.01, reduction='mean', kl=False):
    # error = torch.stop_gradient(torch.abs(y-gamma))
    error = torch.abs(y-gamma)

    if kl:
        kl = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1+omega, beta)
        reg = error*kl
    else:
        evi = 2*v+(alpha)
        reg = error*evi

    if reduction == 'mean':
        return torch.mean(reg, dim=0)
    elif reduction == 'sum':
        return torch.sum(reg)
    else:
        return reg
    # return torch.mean(reg) if reduce else reg

def EvidentialRegression(y_true, evidential_output, coeff=1.0):
    gamma, v, alpha, beta = torch.split(evidential_output, 1, dim=-1)

    loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta) # Maximimizing evidence
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta) # Regularizing evidence

    return loss_nll + coeff * loss_reg
