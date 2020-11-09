import sys
import getopt
import math
import numpy as np
import torch
from torch import nn
from Wp_cost_graph import diffusion_distance
from node2coords import Node2Coords


def main(argv):
    # parse command line arguments
    print('running')
    # first initialize the arguments
    epsilon = 0
    bary_iterations = 0
    latent_dim = 0
    epochs = 0
    batch_size = 0
    rho = 0
    power = 0
    my_tau = 0
    my_p = 0
    dataset = ' '
    device = ''
    try:
        opts, args = getopt.getopt(argv, "e:s:l:i:b:r:d:g:t:p:x:")  # names of arguments must be a single letter
        print(opts, args)
    except getopt.GetoptError:
        print('incorrect arguments')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-e':  # parse argument for entropy regularization
            epsilon = float(arg)  # cast from string to float
        elif opt == '-s':  # parse argument for number of Sinkhorn barycenter iterations
            bary_iterations = int(arg)  # cast from string to integer
        elif opt == '-l':  # parse argument for latent dimensionality
            latent_dim = int(arg)
        elif opt == '-i':   # parse argument for number of epochs
            epochs = int(arg)
        elif opt == '-b':   # parse argument for batch size
            batch_size = int(arg)
        elif opt == '-r':  # parse argument for mass relaxation parameter
            rho = float(arg)
        elif opt == '-d':
            dataset = arg
        elif opt == '-g':
            device = arg
        elif opt == '-t':
            my_tau = int(arg)
        elif opt == '-p':
            my_p = int(arg)
        elif opt == '-x':
            power = int(arg)

    print('The entropic regularization is ', epsilon)
    print('The number of Sinkhorn barycenter iterations is ', bary_iterations)
    print('The latent dimensionality is ', latent_dim)
    print('The mass relaxation parameter is ', rho)
    print('The number of epochs is ', epochs)
    print('The batch size is ', batch_size)
    print('The dataset is ', dataset)
    print('The power of the adjacency matrix is ', power)

    path = './power{}diff{}_tau{}_data_{}_eps{}_sinkh{}_dim{}_epochs{}_batch{}_rho{}/'.format(power, my_p, my_tau, dataset, epsilon, bary_iterations, latent_dim, epochs, batch_size, rho)
    print(path)

    num_nodes = 0

    # normalized adjacency matrix
    adjacency = torch.empty((num_nodes, num_nodes), device=device)
    # cost for mass transportation
    C1 = torch.empty((num_nodes, num_nodes), device=device)

    if dataset == 'karate':
        W = np.load('./data/soc-karate/karate_adjacency.npy')
        num_nodes = W.shape[0]
        WI = W + np.eye(num_nodes)
        Wp = WI
        if power > 1:
            for p in range(power - 1):
                Wp = Wp @ WI
        C, _ = diffusion_distance(W=W, tau=my_tau, p=my_p)
        C = C / np.max(C)
        D = Wp / Wp.sum(axis=0, keepdims=1)
        adjacency = torch.from_numpy(D).type(torch.FloatTensor).to(device)
        C1 = torch.from_numpy(C).type(torch.FloatTensor).to(device)

    elif dataset == 'citeseer4':
        W = np.load('./data/citeseer/adjacency_citeseer_4classes.npy')
        num_nodes = W.shape[0]
        print("The number of nodes is ", num_nodes)
        WI = W + np.eye(num_nodes)
        Wp = WI
        if power > 1:
            for p in range(power - 1):
                Wp = Wp @ WI
        C, _ = diffusion_distance(W=W, tau=my_tau, p=my_p)
        C = C / np.max(C)
        D = Wp / Wp.sum(axis=0, keepdims=1)
        adjacency = torch.from_numpy(D).type(torch.FloatTensor).to(device)
        C1 = torch.from_numpy(C).type(torch.FloatTensor).to(device)

    elif dataset == 'polbooks':
        W = np.load('./data/polbooks/polbooks_adjacency.npy')
        num_nodes = W.shape[0]
        print(num_nodes)
        C, _ = diffusion_distance(W=W, tau=my_tau, p=my_p)
        C = C / np.max(C)
        WI = W + np.eye(num_nodes)
        Wp = WI
        if power > 1:
            for p in range(power - 1):
                Wp = Wp @ WI
        D = Wp / Wp.sum(axis=0, keepdims=1)
        adjacency = torch.from_numpy(D).type(torch.FloatTensor).to(device)
        C1 = torch.from_numpy(C).type(torch.FloatTensor).to(device)

    # Create model
    my_network = Node2Coords(A=adjacency, C1=C1, epsilon=epsilon, rho=rho, N=num_nodes,
                                         S=latent_dim, L=bary_iterations, device=device)

    # Initialize encoder weights
    encoder = torch.empty((num_nodes, latent_dim), requires_grad=True, device=device)
    nn.init.normal_(encoder, mean=0.0, std=1.0)

    # Initialize decoder weights
    bary_weights = torch.empty((num_nodes, latent_dim), requires_grad=True, device=device)
    nn.init.normal_(bary_weights, mean=0.0, std=1.0)

    # create optimizers
    optimizer_encoder = torch.optim.Adam([encoder], lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                         amsgrad=False)
    optimizer_bary_weights = torch.optim.Adam([bary_weights], lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                              amsgrad=False)

    number_of_iterations = math.ceil(num_nodes / batch_size)

    history = []  # to save convergence
    for epoch in range(epochs):
        epoch_loss = 0
        for j in range(number_of_iterations):
            if j == number_of_iterations - 1:
                bary, loss, latent_representation, weight_matrix, my_lamdas = my_network(
                    betas=bary_weights[batch_size * j:num_nodes, :], my_initialization=encoder,
                    real_bary=adjacency[:, batch_size * j:num_nodes],
                    J=adjacency[:, batch_size * j:num_nodes].shape[1])
                loss.backward()  # accumulate gradients for the minibatch
                optimizer_encoder.step()
                optimizer_bary_weights.step()
                optimizer_encoder.zero_grad()
                optimizer_bary_weights.zero_grad()
                epoch_loss = epoch_loss + loss.item()
                if (epoch+1) == epochs:
                    if (j+1) == number_of_iterations:
                        torch.save(latent_representation,
                                   path + 'encoder/latent_rep/latent_representation_epoch{}.pt'.format(epoch), _use_new_zipfile_serialization=False)
                    torch.save(my_lamdas, path + 'weights/lamdas_it{}_epoch{}.pt'.format(j, epoch), _use_new_zipfile_serialization=False)

            else:
                bary, loss, latent_representation, weight_matrix, my_lamdas = my_network(
                    betas=bary_weights[batch_size * j:batch_size * (j + 1), :], my_initialization=encoder,
                    real_bary=adjacency[:, batch_size * j:batch_size * (j + 1)],
                    J=batch_size)
                loss.backward()  # accumulate gradients for the minibatch
                optimizer_encoder.step()
                optimizer_bary_weights.step()
                optimizer_encoder.zero_grad()
                optimizer_bary_weights.zero_grad()
                epoch_loss = epoch_loss + loss.item()
                if (epoch + 1) == epochs:
                    if (j + 1) == number_of_iterations:
                        torch.save(latent_representation,
                                   path + 'encoder/latent_rep/latent_representation_epoch{}.pt'.format(epoch), _use_new_zipfile_serialization=False)
                    torch.save(my_lamdas, path + 'weights/lamdas_it{}_epoch{}.pt'.format(j, epoch), _use_new_zipfile_serialization=False)

        avg_epoch_loss = epoch_loss / number_of_iterations
        print("The loss at epoch {} is {}".format(epoch, avg_epoch_loss))
        history.append(avg_epoch_loss)
        if epoch == epochs - 1:
            my_history = np.asarray(history)
            np.save(path + 'convergence', my_history)

    print("Optimization Finished!")


if __name__ == "__main__":
    main(sys.argv[1:])


