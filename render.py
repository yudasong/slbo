# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import pickle
from collections import deque
import tensorflow as tf
import numpy as np
from tqdm import tqdm

import lunzi.nn as nn
from lunzi.Logger import logger
from slbo.utils.average_meter import AverageMeter
from slbo.utils.flags import FLAGS
from slbo.utils.dataset import Dataset, gen_dtype
from slbo.utils.OU_noise import OUNoise
from slbo.utils.normalizer import Normalizers
from slbo.utils.tf_utils import get_tf_config
from slbo.utils.runner import Runner
from slbo.policies.gaussian_mlp_policy import GaussianMLPPolicy
from slbo.envs.virtual_env import VirtualEnv
from slbo.dynamics_model import DynamicsModel
from slbo.v_function.mlp_v_function import MLPVFunction
from slbo.partial_envs import make_env
from slbo.loss.multi_step_loss import MultiStepLoss
from slbo.algos.TRPO import TRPO
from slbo.random_net import RandomNet


def evaluate(settings, tag):
    return_means = []
    for runner, policy, name in settings:
        runner.reset()
        _, ep_infos = runner.run(policy, FLAGS.rollout.n_test_samples)
        if name == 'Real Env':
            returns = np.array([ep_info['success'] for ep_info in ep_infos])
        else:
            returns = np.array([ep_info['return'] for ep_info in ep_infos])
        logger.info('Tag = %s, Reward on %s (%d episodes): mean = %.6f, std = %.6f', tag, name,
                    len(returns), np.mean(returns), np.std(returns))

        return_means.append(np.mean(returns))

    return return_means



def add_multi_step(src: Dataset, dst: Dataset):
    n_envs = 1
    dst.extend(src[:-n_envs])

    ending = src[-n_envs:].copy()
    ending.timeout = True
    dst.extend(ending)


def make_real_runner(n_envs):
    from slbo.envs.batched_env import BatchedEnv
    batched_env = BatchedEnv([make_env(FLAGS.env.id) for _ in range(n_envs)])
    return Runner(batched_env, rescale_action=True, **FLAGS.runner.as_dict())


def main():
    FLAGS.set_seed()
    FLAGS.freeze()

    env = make_env(FLAGS.env.id)
    dim_state = int(np.prod(env.observation_space.shape))
    dim_action = int(np.prod(env.action_space.shape))

    #env.verify()

    normalizers = Normalizers(dim_action=dim_action, dim_state=dim_state)

    dtype = gen_dtype(env, 'state action next_state reward done timeout')
    train_set = Dataset(dtype, FLAGS.rollout.max_buf_size)
    dev_set = Dataset(dtype, FLAGS.rollout.max_buf_size)

    policy = GaussianMLPPolicy(dim_state, dim_action, normalizer=normalizers.state, **FLAGS.policy.as_dict())
    # batched noises
    #noise = OUNoise(env.action_space, theta=FLAGS.OUNoise.theta, sigma=FLAGS.OUNoise.sigma, shape=(1, dim_action))
    vfn = MLPVFunction(dim_state, [64, 64], normalizers.state)
    model = DynamicsModel(dim_state, dim_action, normalizers, FLAGS.model.hidden_sizes)
    random_net = RandomNet(dim_state, dim_action, normalizers, FLAGS.model.hidden_sizes)

    virt_env = VirtualEnv(model, make_env(FLAGS.env.id), random_net, FLAGS.plan.n_envs,FLAGS.model.hidden_sizes[-1], 
                            FLAGS.pc.bonus_scale,FLAGS.pc.lamb, opt_model=FLAGS.slbo.opt_model)
    virt_runner = Runner(virt_env, **{**FLAGS.runner.as_dict(), 'max_steps': FLAGS.plan.max_steps})

    criterion_map = {
        'L1': nn.L1Loss(),
        'L2': nn.L2Loss(),
        'MSE': nn.MSELoss(),
    }
    criterion = criterion_map[FLAGS.model.loss]
    loss_mod = MultiStepLoss(model, normalizers, dim_state, dim_action, criterion, FLAGS.model.multi_step)
    loss_mod.build_backward(FLAGS.model.lr, FLAGS.model.weight_decay)
    algo = TRPO(vfn=vfn, policy=policy, dim_state=dim_state, dim_action=dim_action, **FLAGS.TRPO.as_dict())

    tf.get_default_session().run(tf.global_variables_initializer())

    runners = {
        'test': make_real_runner(4),
        'collect': make_real_runner(1),
        'dev': make_real_runner(1),
        'train': make_real_runner(FLAGS.plan.n_envs) if FLAGS.algorithm == 'MF' else virt_runner,
    }
    settings = [(runners['test'], policy, 'Real Env'), (runners['train'], policy, 'Virt Env')]

    saver = nn.ModuleDict({'policy': policy, 'model': model, 'vfn': vfn})
    print(saver)

    eval_real_returns = []
    timesteps = []

    if FLAGS.ckpt.model_load:
        saver.load_state_dict(np.load(FLAGS.ckpt.model_load, allow_pickle=True)[()])
        logger.warning('Load model from %s', FLAGS.ckpt.model_load)

    if FLAGS.ckpt.buf_load:
        n_samples = 0
        for i in range(FLAGS.ckpt.buf_load_index):
            data = pickle.load(open(f'{FLAGS.ckpt.buf_load}/stage-{i}.inc-buf.pkl', 'rb'))
            add_multi_step(data, train_set)
            n_samples += len(data)
        logger.warning('Loading %d samples from %s', n_samples, FLAGS.ckpt.buf_load)

    max_ent_coef = 0
    
    recent_train_set, ep_infos = runners['collect'].run(policy, FLAGS.rollout.n_train_samples, render=FLAGS.ckpt.render)

    returns = np.array([ep_info['success'] for ep_info in ep_infos])

    if len(returns) > 0:
        logger.info("episode: %s", np.mean(returns))


if __name__ == '__main__':
    with tf.Session(config=get_tf_config()):
        main()
