import torch
import numpy as np
import copy
import os
import scipy.spatial
import scipy.signal


def median_trick(X, args):
    #median trick for computing the bandwith for kernel regression.
    N = X.shape[0]
    #print(X.shape)
    perm = np.random.choice(N, np.min([N,args.update_size * args.buffer_width]), replace=False)
    dsample = X[perm]
    pd = scipy.spatial.distance.pdist(dsample)
    sigma = np.median(pd)
    return sigma

def compute_cov_pi(phi):
    #cov = np.zeros((phi.shape[1],phi.shape[1]))

    #for i in range(len(phi)):
    #    cov += np.outer(phi[i],phi[i])
    cov = np.dot(phi.T,phi)
    cov /= phi.shape[0]

    #print(cov)

    return cov

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]



def denormalize(ob_rms, state):
    var = ob_rms.var
    #print(var)
    return torch.FloatTensor(state.data.numpy() * np.sqrt(var) + ob_rms.mean)

def normalize(ob_rms, state):
    return torch.FloatTensor((state.data.numpy() - ob_rms.mean) / np.sqrt(ob_rms.var))

def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def save(dynamics, timestep_list, reward_list, bonus_list, loss_list, cov, sigma, args):

    save_path = "trained_models/{}/{}_{}_{}_{}".format(args.env_name, args.phi_dim, args.bonus_scale, args.lamb, args.sample_size, args.plan_horizen)
    if args.use_v_net:
        save_path += "_vnet"
    if args.normalize:
        save_path += "_norm"
    if args.no_bonus:
        save_path += "_no_bonus"
    try:
        os.makedirs(save_path)
    except OSError:
        pass
    torch.save(
        dynamics, os.path.join(
        save_path ,
        "{}.pt".format(str(args.seed))))

    np.save(
        os.path.join(
        save_path ,
        "cov_{}".format(str(args.seed))),
        [cov,sigma])


    save_path = "data/{}/{}_{}_{}_{}_{}_{}".format(args.env_name, 
        args.phi_dim, args.bonus_scale, args.lamb, args.buffer_width, 
        args.plan_horizen, args.lr, args.v_lr)
    if args.use_v_net:
        save_path += "_vnet"
    if args.normalize:
        save_path += "_norm"
    if args.no_bonus:
        save_path += "_no_bonus"
    try:
        os.makedirs(save_path)
    except OSError:
        pass


    np.save(save_path+"/ts_{}".format(str(args.seed)),timestep_list)
    np.save(save_path+"/rw_{}".format(str(args.seed)),reward_list)
    np.save(save_path+"/bonus_{}".format(str(args.seed)),bonus_list)
    np.save(save_path+"/loss_{}".format(str(args.seed)),loss_list)
