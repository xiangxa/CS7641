"""
Course:         CS 7641 Assignment 4, Spring 2020
Date:           March 31st, 2020
Author:         Maziar Mardan
GT Username:    mmardan3
"""

from hiive.mdptoolbox import mdp

from Forest import *

import matplotlib.pyplot as plt
import numpy as np
import QLearner




def findBestPolicyForForest():
	cntr = 0
	pi_rewards = []
	pi_error = []
	pi_time = []
	pi_iter = []
	vi_rewards = []
	vi_error = []
	vi_time = []
	vi_iter = []
	for size in [1000]:
		forest = ForestMng(states=size, reward_wait=4, reward_cut=2, prob_fire=0.3)

		# Policy iteration
		policy_iteration = mdp.PolicyIteration(forest.P, forest.R, gamma=0.9, policy0=None, max_iter=1000, eval_type=0)
		policy_iteration.run()
		print(policy_iteration.time)
		print(policy_iteration.iter)
		print(policy_iteration.policy)
		pi_rewards.append([sub['Reward'] for sub in policy_iteration.run_stats])
		pi_error.append([ sub['Error'] for sub in policy_iteration.run_stats ])
		pi_time.append([ sub['Time'] for sub in policy_iteration.run_stats ])
		pi_iter.append([ sub['Iteration'] for sub in policy_iteration.run_stats ])

		# Value iteration
		value_iteration = mdp.ValueIteration(forest.P, forest.R, gamma=0.9, max_iter=1000)
		value_iteration.run()
		print(value_iteration.time)
		print(value_iteration.iter)
		print(value_iteration.policy)
		vi_rewards.append([sub['Reward'] for sub in value_iteration.run_stats])
		vi_error.append([sub['Error'] for sub in value_iteration.run_stats])
		vi_time.append([sub['Time'] for sub in value_iteration.run_stats])
		vi_iter.append([sub['Iteration'] for sub in value_iteration.run_stats])

		if max(pi_iter[cntr]) < max(vi_iter[cntr]):
			for i in range(max(vi_iter[cntr]) - max(pi_iter[cntr])):
				pi_error[cntr].append(pi_error[cntr][len(pi_error[cntr])-1])
				pi_rewards[cntr].append(pi_rewards[cntr][len(pi_rewards[cntr]) - 1])
				pi_time[cntr].append(pi_time[cntr][len(pi_time[cntr]) - 1])

		cntr += 1

	plt.style.use('seaborn-whitegrid')
	plt.plot(vi_iter[0], pi_error[0], label='PI')
	plt.plot(vi_iter[0], vi_error[0], label='VI')
	plt.ylabel('Convergence', fontsize=12)
	plt.xlabel('Iter.', fontsize=12)
	plt.title('Convergence of Error vs Iteration for Forest Mng State 1000 p03', fontsize=12, y=1.03)
	plt.legend()
	plt.savefig('Figures/Forest/Error Convergence vs Iteration for Forest Mng state 1000 p03.png')
	plt.close()

	plt.style.use('seaborn-whitegrid')
	plt.plot(vi_iter[0], pi_rewards[0], label='PI')
	plt.plot(vi_iter[0], vi_rewards[0], label='VI')
	plt.ylabel('Reward', fontsize=12)
	plt.xlabel('Iter.', fontsize=12)
	plt.title('Rewards vs Iteration for Forest Mng state 1000 p03', fontsize=12, y=1.03)
	plt.legend()
	plt.savefig('Figures/Forest/Rewards vs Iteration for Forest Mng state 1000 p03.png')
	plt.close()

	plt.style.use('seaborn-whitegrid')
	plt.plot(vi_iter[0], pi_time[0], label='PI')
	plt.plot(vi_iter[0], vi_time[0], label='VI')
	plt.ylabel('Time', fontsize=12)
	plt.xlabel('Iter.', fontsize=12)
	plt.title('Time vs Iteration for Forest Mng state 1000 p03', fontsize=12, y=1.03)
	plt.legend()
	plt.savefig('Figures/Forest/Time vs Iteration for Forest Mng  state 1000 p3.png')
	plt.close()


def getPlotsForForestQl():
	iters = range(1, 21, 1)
	lRates = [x for x in [0.8, 0.9]]
	epsilons = [x for x in [0.8, 0.9]]
	ql_rewards = []
	ql_error = []
	ql_time = []
	ql_iter = []

	forest = ForestMng(states=1000, reward_wait=4, reward_cut=2 prob_fire=0.3)

	for lRate in lRates:
		for epsilon in epsilons:
			# Q-Learning
			q_learning = QLearner.QLearningEx(forest.P, forest.R, grid=np.zeros(shape=(15, 1)), start=0, goals=[14],
											  n_iter=1000, n_restarts=1000, alpha=lRate, gamma=0.9, rar=epsilon,
											  radr=0.999999)
			q_learning.run()
			ql_rewards.append(q_learning.episode_reward)
			ql_time.append(q_learning.episode_times)
			ql_error.append(q_learning.episode_error)
			print(q_learning.policy)

	elCntr = 0

	print("First Combination reward mean: ", np.mean(ql_rewards[0]))
	print("Second Combination reward mean: ", np.mean(ql_rewards[1]))
	print("Third Combination reward mean: ", np.mean(ql_rewards[2]))
	print("Four Combination reward mean: ", np.mean(ql_rewards[3]))
	print("First Combination error mean: ", np.mean(ql_error[0]))
	print("Second Combination error mean: ", np.mean(ql_error[1]))
	print("Third Combination error mean: ", np.mean(ql_error[2]))
	print("Four Combination error mean: ", np.mean(ql_error[3]))

	plt.figure(figsize=(15, 8))
	plt.style.use('seaborn-whitegrid')
	for lRate in lRates:
		for epsilon in epsilons:
			if lRate == 0.8:
				plt.plot(range(0, 1000)[::10], ql_error[elCntr][::10],
						 label='a: ' + str(lRate) + ', e: ' + str(epsilon))
				elCntr += 1
			else:
				plt.plot(range(0, 1000)[::10], ql_error[elCntr][::10],
						 label='a: ' + str(lRate) + ', e: ' + str(epsilon), linestyle='--')
				elCntr += 1
	plt.ylabel('Convergence', fontsize=12)
	plt.xlabel('Iter.', fontsize=12)
	plt.title('Error Convergence vs Iteration for Forest Mng State 1000 fire = 0.3', fontsize=12, y=1.03)
	plt.legend()
	plt.savefig('Figures/Forest/Convergence vs Iteration for Forest Mng, QL State 1000.png')
	plt.close()

	elCntr = 0

	plt.figure(figsize=(15, 8))
	plt.style.use('seaborn-whitegrid')
	for lRate in lRates:
		for epsilon in epsilons:
			if lRate == 0.8:
				plt.plot(range(0, 1000)[::10], ql_rewards[elCntr][::10],
						 label='a: ' + str(lRate) + ', e: ' + str(epsilon))
			else:
				plt.plot(range(0, 1000)[::10], ql_rewards[elCntr][::10],
						 label='a: ' + str(lRate) + ', e: ' + str(epsilon), linestyle='--')
			elCntr += 1
	plt.ylabel('Reward', fontsize=12)
	plt.xlabel('Iter.', fontsize=12)
	plt.title('Reward vs Iteration for Forest Mng state 1000', fontsize=12, y=1.03)
	plt.legend()
	plt.savefig('Figures/Forest/Reward vs Iteration for Forest Mng, QL State 1000.png')
	plt.close()


def main():
	findBestPolicyForForest()
	getPlotsForForestQl()


if __name__ == '__main__':
	main()
