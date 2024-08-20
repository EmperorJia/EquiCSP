from torch_scatter import scatter
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import norm
import numpy as np
from tqdm import trange
from scipy.stats import vonmises
from scipy.optimize import minimize

def calc_grouped_angles_mean_in_radians(data_tensor, groups):
    data_tensor = data_tensor * 2*math.pi

    data_tensor_sin = torch.sin(data_tensor)
    data_tensor_cos = torch.cos(data_tensor)

    sum_sin = scatter(data_tensor_sin, groups, dim=0, reduce='sum')
    sum_cos = scatter(data_tensor_cos, groups, dim=0, reduce='sum')

    group_counts = scatter(torch.ones_like(data_tensor), groups, dim=0, reduce='sum')

    mean_sin = sum_sin / group_counts
    mean_cos = sum_cos / group_counts

    mean_angle = torch.atan2(mean_sin, mean_cos)  # Calculate mean angle in radians using atan2 for stability

    # Adjust mean_angle to be in the range [0, 2*pi)
    mean_angle = torch.where(mean_angle >= 0, mean_angle, mean_angle + 2 * math.pi)

    mean_angle = mean_angle / (2*math.pi)

    return mean_angle

def get_trans_x(n, mu=0.0, sigma=0.1):
    sample_size = 10000
    sample_size = int(sample_size / n)
    samples = np.random.normal(mu, sigma, sample_size*n)
    wrapped_samples = np.mod(samples, 1)
    rand_x = torch.tensor(wrapped_samples)
    invarance_reference = calc_grouped_angles_mean_in_radians(rand_x.view(-1, 1), torch.tensor([i for i in range(sample_size) for _ in range(n)]))
    invarance_reference = invarance_reference.repeat_interleave(n, dim=0)
    invariance_noise = ((rand_x.view(-1, 1) - invarance_reference)%1.).view(-1)
    return invariance_noise.numpy()

def neg_log_likelihood(params, data):
    kappa = params
    return -np.sum(vonmises.logpdf(data, kappa, loc=0.0, ))

def sample_norm(sigma, T=1.0, sn = 10000, num_atoms=52):
    sn_list = []
    print('Monte Carlo calculating sample norm')
    for _ in trange(sn):
        num_list = []
        for n in range(1, num_atoms+1):
            sigmas = sigma[None, :].repeat(n, 1)
            x_sample = sigmas * torch.randn_like(sigmas)
            x_sample = x_sample % T
            x_sample_angle = x_sample * 2*math.pi
            x_sample_sin = torch.sin(x_sample_angle)
            x_sample_cos = torch.cos(x_sample_angle)
            mean_sin = x_sample_sin.mean(dim=0)
            mean_cos = x_sample_cos.mean(dim=0)
            mean_angle = torch.atan2(mean_sin, mean_cos)
            mean_angle = torch.where(mean_angle >= 0, mean_angle, mean_angle + 2 * math.pi)
            mean_angle = (mean_angle / (2*math.pi))[None, :]
            x_sample = (x_sample - mean_angle) % T
            normal_ = torch.where(x_sample<=0.5, x_sample, x_sample-1.0)
            normal_ = normal_ ** 2
            normal_ = normal_.mean(dim=0).view(-1)
            num_list.append(normal_)
        num_list = torch.stack(num_list, dim=0)
        sn_list.append(num_list)
    sn_list = torch.stack(sn_list, dim=0)
    print('finish')
    return (sn_list).mean(dim = 0)

if __name__=='__main__':
    sigmas = np.exp(np.linspace(np.log(0.005), np.log(0.5), 1000))
    max_atoms = 52
    kappa_matrix = np.zeros((max_atoms+1, 1000), dtype=float)
    kappa_norm = np.zeros((max_atoms+1, 1000), dtype=float)
    for n in range(2, max_atoms+1):
        print('process n: ', n)
        for i in trange(len(sigmas)):
            sigma = sigmas[i]
            # Generate sample data from a normal distribution and wrap it into [0, 1]

            data = get_trans_x(n, 0.0, sigma)

            data = data * 2*np.pi
            initial_guess = [sigma]
            bounds = [(1e-6, 10000)]
            result = minimize(neg_log_likelihood, initial_guess, bounds=bounds, args=(data,))
            kappa = result.x[0]
            kappa_matrix[n][i] = kappa

            # pdf_values = 2*np.pi*vonmises.pdf(x_grid*2*np.pi, kappa=result.x[0], loc=0)

            d_log_data_expect = kappa*np.sin(data)*2*np.pi
            kappa_norm[n][i] = (d_log_data_expect**2).mean()

        torch.save(kappa_matrix, './equicsp/normalization/kappa_matrix.pth')
        torch.save(kappa_norm, './equicsp/normalization/kappa_norm.pth')
    
    sigmas = torch.FloatTensor(np.exp(np.linspace(np.log(0.005), np.log(0.5), 1000))).to('cuda')
    sample_norm_ = sample_norm(sigmas, num_atoms=52)
    torch.save(sample_norm_, './equicsp/normalization/sample_norm.pth')

