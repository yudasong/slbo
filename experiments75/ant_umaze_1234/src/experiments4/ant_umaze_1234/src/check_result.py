import numpy as np

seed_list = ['1234', '1235', '2314', '2345']
env_name = 'gym_reacher'
lasts = []
for seed in seed_list:
    cur = np.load('experiments/'+env_name+'_'+seed+'/eval_real_returns.npy')
    lasts.append(cur[-1])

print(lasts)
print(np.mean(lasts))
print(np.std(lasts))
