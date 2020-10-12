# -*- coding: utf-8 -*-

import mlrose_hiive as mlrose
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from time import process_time

print("Running OneMax...")

fitness = mlrose.fourpeaks()
problem = mlrose.DiscreteOpt(100, fitness)

RANDOM_SEED = 42
MAX_ATTEMPTS = 200

#%% tuning for SA
curve_list = []
decays = [0.999, 0.99, 0.9,0.8,0.5]
for d in decays:
    schedule = mlrose.GeomDecay(decay=d)
    _, _, curve = mlrose.simulated_annealing(
        problem,
        schedule=schedule,
        max_attempts=MAX_ATTEMPTS,
        max_iters=500,
        curve=True,
        random_state=RANDOM_SEED,
    )
    curve_list.append(curve)

df = pd.DataFrame(curve_list).transpose()
df.columns = decays
df.plot()
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.title("OneMax: Fitness curve vs decay rate in SA")
plt.savefig("output/onemax_sa_decay.png")
plt.close()

#%% tuning for GA
curve_list = []
pop_sizes = [10, 20, 30, 40, 100, 200]
for p in pop_sizes:
    _, _, curve = mlrose.genetic_alg(
        problem,
        max_attempts=MAX_ATTEMPTS,
        max_iters=100,
        pop_size=p,
        elite_dreg_ratio=1,
        curve=True,
        random_state=RANDOM_SEED,
    )
    curve_list.append(curve)

df = pd.DataFrame(curve_list).transpose()
df.columns = pop_sizes
df.plot()
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.title("OneMax: Fitness curve vs population size in GA")
plt.savefig("output/onemax_ga_pop.png")
plt.close()

#%% tuning for RHC
curve_list = []
max_iters = [10, 20, 30, 40, 100]
for m in max_iters:
    _, _, curve = mlrose.random_hill_climb(
        problem, max_attempts=MAX_ATTEMPTS, max_iters = m
        , curve=True, random_state=RANDOM_SEED
    )
    curve_list.append(curve)

df = pd.DataFrame(curve_list).transpose()
df.columns = max_iters
df.plot()
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.title("OneMax: Fitness curve vs max iter in RHC")
plt.savefig("output/onemax_RHC_maxiter.png")
plt.close()



#%% tuning for MIMIC

curve_list = []
pop_sizes = [50, 100, 200]
for p in pop_sizes:
    _, _, curve = mlrose.mimic(
        problem,
        max_attempts=MAX_ATTEMPTS,
        max_iters=50,
        pop_size=p,
        curve=True,
        random_state=RANDOM_SEED,
    )
    curve_list.append(curve)

df = pd.DataFrame(curve_list).transpose()
df.columns = pop_sizes
df.plot()
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.title("OneMax: Fitness curve vs population size in MIMIC")
plt.savefig("output/onemax_mimic_pop.png")
plt.close()

#%% Putting together
RANDOM_SEED = 21

curve_list = []
time_list = []
n_eval = []
algo_list = ["RHC", "SA", "GA", "MIMIC"]

# RHC
t1 = process_time()
_, _, curve = mlrose.random_hill_climb(
    problem, max_attempts=MAX_ATTEMPTS, curve=True, random_state=RANDOM_SEED
)
t2 = process_time()
time_list.append((t2 - t1) / len(curve))
curve_list.append(curve)
n_eval.append(np.argmax(curve) + 1)

# SA
schedule = mlrose.GeomDecay(decay=0.9)
t1 = process_time()
_, _, curve = mlrose.simulated_annealing(
    problem,
    schedule=schedule,
    max_attempts=MAX_ATTEMPTS,
    curve=True,
    random_state=RANDOM_SEED,
)
t2 = process_time()
time_list.append((t2 - t1) / len(curve))
curve_list.append(curve)
n_eval.append(np.argmax(curve) + 1)

# GA
t1 = process_time()
_, _, curve = mlrose.genetic_alg(
    problem,
    max_attempts=MAX_ATTEMPTS,
    max_iters=100,
    pop_size=100,
    elite_dreg_ratio=1,
    curve=True,
    random_state=RANDOM_SEED,
)
t2 = process_time()
time_list.append((t2 - t1) / len(curve))
curve_list.append(curve)
n_eval.append((np.argmax(curve) + 1) * 100)

# MIMIC
t1 = process_time()
_, _, curve = mlrose.mimic(
    problem,
    max_attempts=MAX_ATTEMPTS,
    max_iters=50,
    curve=True,
    random_state=RANDOM_SEED,
)
t2 = process_time()
time_list.append((t2 - t1) / len(curve))
curve_list.append(curve)
n_eval.append((np.argmax(curve) + 1) * 200)

df = pd.DataFrame(curve_list).transpose()
df.columns = algo_list
df.plot()
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.title("OneMax: Fitness curve vs algorithms")
plt.savefig("output/onemax_algo.png")
plt.close()

print("time per iteration:")
print(time_list)
print("number of func eval reaching maxima:")
print(n_eval)
