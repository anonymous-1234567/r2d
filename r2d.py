import torch
import math
import numpy as np
from math import exp, sqrt
from scipy.special import erf


def h_function(K, eta, L, m, n, T):
    return(((1 + eta * L * n/(n-m))**(T-K) - 1) * (1 + eta * L)**K)


def add_gaussian_noise_to_weights(model, sigma, device, seed=None):

    if seed is not None:
        torch.manual_seed(seed)
    # Iterate over all model parameters
    for param in model.parameters():
        # Check if the parameter has a gradient (usually it will if it's a learnable weight)
        if param.requires_grad:
            # Create Gaussian noise with mean 0 and standard deviation sigma
            noise = torch.normal(mean=0.0, std=sigma, size=param.size())
            # Add the noise to the parameter
            param.data += noise.to(device)


def get_error(output, target):
    if output.shape[1]>1:
        pred = output.argmax(dim=1, keepdim=True)
        return (1. - pred.eq(target.view_as(pred)).float().mean().item(), len(pred))


def compute_test_error(model, loader, device=None):
    if device is None:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        
    model.eval()
    with torch.no_grad():
        
        error_list = []

        for batch_idx, (data, target, identity) in enumerate(loader):
                    #data, target = data.to(args.device), target.to(args.device)   
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            error, length = get_error(output, target)

            error_list.append(error * length)

        return(sum(error_list)/len(loader.dataset))


def calibrateAnalyticGaussianMechanism(epsilon, delta, GS, tol = 1.e-12):
    #https://github.com/Anonymous202401/If-Recollecting-were-Forgetting/blob/main/utils/perturbation.py
    """ Calibrate a Gaussian perturbation for differential privacy using the analytic Gaussian mechanism of [Balle and Wang, ICML'18]

    Arguments:
    epsilon : target epsilon (epsilon > 0)
    delta : target delta (0 < delta < 1)
    GS : upper bound on L2 global sensitivity (GS >= 0)
    tol : error tolerance for binary search (tol > 0)

    Output:
    sigma : standard deviation of Gaussian noise needed to achieve (epsilon,delta)-DP under global sensitivity GS
    """

    def Phi(t):
        return 0.5*(1.0 + erf(float(t)/sqrt(2.0)))

    def caseA(epsilon,s):
        return Phi(sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def caseB(epsilon,s):
        return Phi(-sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def doubling_trick(predicate_stop, s_inf, s_sup):
        while(not predicate_stop(s_sup)):
            s_inf = s_sup
            s_sup = 2.0*s_inf
        return s_inf, s_sup

    def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup-s_inf)/2.0
        while(not predicate_stop(s_mid)):
            if (predicate_left(s_mid)):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup-s_inf)/2.0
        return s_mid

    delta_thr = caseA(epsilon, 0.0)

    if (delta == delta_thr):
        alpha = 1.0

    else:
        if (delta > delta_thr):
            predicate_stop_DT = lambda s : caseA(epsilon, s) >= delta
            function_s_to_delta = lambda s : caseA(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) > delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) - sqrt(s/2.0)

        else:
            predicate_stop_DT = lambda s : caseB(epsilon, s) <= delta
            function_s_to_delta = lambda s : caseB(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) < delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) + sqrt(s/2.0)

        predicate_stop_BS = lambda s : abs(function_s_to_delta(s) - delta) <= tol

        s_inf, s_sup = doubling_trick(predicate_stop_DT, 0.0, 1.0)
        s_final = binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
        alpha = function_s_to_alpha(s_final)
        
    sigma = alpha*GS/sqrt(2.0*epsilon)

    return sigma