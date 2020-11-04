import torch
import torch.nn as nn


# define the change of variable layer for barycentric coordinates
class DeltasToLamdas(nn.Module):
    def __init__(self, device):
        super(DeltasToLamdas, self).__init__()
        self.device = device

    def forward(self, betas):
        exp_betas = torch.exp(betas)
        denominator = torch.sum(exp_betas, 1)
        normalized = torch.div(torch.t(exp_betas), denominator)
        return torch.t(normalized)


class WassersteinBarycenterIteration(nn.Module):
    def __init__(self, kernel, epsilon, rho, dim, S, device):
        super(WassersteinBarycenterIteration, self).__init__()
        self.K = nn.Parameter(kernel)  # Gibbs kernel for mass transportation
        self.N = dim  # number of nodes
        self.S = S  # number of marginals
        self.epsilon = epsilon  # entropy regularization parameter
        self.rho = rho  # mass relaxation parameter
        self.exp = self.rho / (self.rho + self.epsilon)
        self.bary_exp = self.epsilon / (self.rho + self.epsilon)
        self.device = device

    def forward(self, u, v, lamdas, marginals):
        # update first scaling vectors
        u1 = torch.pow(torch.div(marginals, torch.matmul(self.K, v)+1e-19), self.exp)

        # estimate the barycenter
        temp1 = v * torch.matmul(self.K, u1)
        temp2 = torch.pow(temp1, self.bary_exp)
        temp3 = temp2.permute(1, 0, 2)
        temp4 = torch.sum(lamdas[None, :] * temp3, axis=2)
        bary_1 = torch.pow(temp4, 1/self.bary_exp)

        # update second scaling vectors
        Ku1 = torch.matmul(self.K, u1)
        Ku1_temp = Ku1.permute(2, 1, 0)
        v1_temp = torch.div(bary_1, Ku1_temp+1e-19)
        v1 = torch.pow(v1_temp.permute(2, 1, 0), self.exp)
        return bary_1, u1, v1


# define the autoencoder network node2coords
class Node2Coords(nn.Module):
    def __init__(self, A, C1, epsilon, rho, N, S, L, device):
        super(Node2Coords, self).__init__()
        self.device = device
        self.A = nn.Parameter(A)  # adjacency matrix of the graph
        self.C1 = nn.Parameter(C1)  # cost for barycenter computation
        self.epsilon = epsilon  # entropy regularization parameter
        self.K1 = torch.exp(-self.C1 / self.epsilon)  # Gibbs kernel
        self.rho = rho  # mass relaxation parameter
        self.N = N  # dimensionality of histograms (aka signals)
        self.S = S  # dimensionality of latent space
        self.L = L  # number of Sinkhorn layers
        self.d_to_l = DeltasToLamdas(self.device)
        self.bary = WassersteinBarycenterIteration(kernel=self.K1, epsilon=self.epsilon, rho=self.rho, dim=self.N, S=self.S, device=self.device)

    def forward(self, betas, my_initialization, real_bary, J):
        #  ENCODER  #
        lin_comb = torch.mm(self.A, my_initialization)
        latent_representation = nn.functional.softmax(lin_comb, dim=0)

        #  DECODER  #
        # change of variable for barycentric weights
        my_lamdas = self.d_to_l(betas)
        # initialization of scaling vectors
        u1_init = torch.ones(J, self.N, self.S).to(self.device)
        v1_init = torch.ones(J, self.N, self.S).to(self.device)

        # first Sinkhorn iteration
        bary, u1, v1 = self.bary(u1_init, v1_init, my_lamdas, latent_representation)

        # Sinkhorn iterations
        for i in range(self.L-1):
            bary, u1, v1 = self.bary(u1, v1, my_lamdas, latent_representation)

        # loss computation
        loss = torch.mean(torch.pow((bary - real_bary), 2).sum(0))
        norms = torch.pow(real_bary, 2).sum(0)
        loss_norm = loss / torch.mean(norms)

        return bary, loss_norm, latent_representation, my_initialization, my_lamdas
