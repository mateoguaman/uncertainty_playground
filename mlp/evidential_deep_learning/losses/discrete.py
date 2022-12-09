# Not used right now

# import torch
# import numpy as np

# def Dirichlet_SOS(y, alpha, t):
#     def KL(alpha):
#         beta=torch(np.ones((1,alpha.shape[1])),dtype=torch.float32)
#         S_alpha = torch.sum(alpha,axis=1,keepdims=True)
#         S_beta = torch.sum(beta,axis=1,keepdims=True)
#         lnB = torch.math.lgamma(S_alpha) - torch.sum(torch.math.lgamma(alpha),axis=1,keepdims=True)
#         lnB_uni = torch.sum(torch.math.lgamma(beta),axis=1,keepdims=True) - torch.math.lgamma(S_beta)
#         lnB_uni = torch.sum(torch.math.lgamma(beta),axis=1,keepdims=True) - torch.math.lgamma(S_beta)

#         dg0 = torch.math.digamma(S_alpha)
#         dg1 = torch.math.digamma(alpha)

#         kl = torch.sum((alpha - beta)*(dg1-dg0),axis=1,keepdims=True) + lnB + lnB_uni
#         return kl

#     S = torch.sum(alpha, axis=1, keepdims=True)
#     evidence = alpha - 1
#     m = alpha / S

#     A = torch.sum((y-m)**2, axis=1, keepdims=True)
#     B = torch.sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keepdims=True)

#     # annealing_coef = torch.minimum(1.0,torch.cast(global_step/annealing_step,torch.float32))
#     alpha_hat = y + (1-y)*alpha
#     C = KL(alpha_hat)

#     C = torch.mean(C, axis=1)
#     return torch.mean(A + B + C)

# def Sigmoid_CE(y, y_logits):
#     loss = torch.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_logits)
#     return torch.mean(loss)
