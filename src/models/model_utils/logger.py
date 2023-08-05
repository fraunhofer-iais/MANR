import numpy as np
import argparse
import sys
import os
import csv
import re
import json
import pandas as pd
import pdb
import pickle

import pyarrow.feather as feather


class Logger(object):
	"""
	The class for recording the training process.
	"""
	# relative paths don't work with ray!
	def __init__(self, args):
		# only training info logged
		# experiment_name
		self.log_interval = args.log_interval
		# self.log_name = "../logs/" + args.experiment_name + "/central_vrp0.txt"
		self.model_saving_dir = "/home/IAIS/npaul/comb_opt/pycharm_sync_marl_neural_rewriter/logs/" + args.experiment_name
		self.log_name_part = "/home/IAIS/npaul/comb_opt/pycharm_sync_marl_neural_rewriter/logs/" + args.experiment_name + "/central_vrp_"
		#if not os.path.exists(f"../logs/{args.experiment_name}"):
		if not os.path.exists(f"/home/IAIS/npaul/comb_opt/pycharm_sync_marl_neural_rewriter/logs/{args.experiment_name}"):
			#os.makedirs(f"../logs/{args.experiment_name}")
			os.makedirs(f"/home/IAIS/npaul/comb_opt/pycharm_sync_marl_neural_rewriter/logs/{args.experiment_name}")


	def save_init_nodes(self, init_nodes):
		# save df to fearther (shall be more efficient than pickle
		print("Saving distribution of init nodes")
		print(f"to {self.model_saving_dir}")
		file_name = self.model_saving_dir+"/init_nodes_eval.p"
		# save init route nodes
		pickle.dump(init_nodes, open(file_name, "wb"))


	def save_df_results(self, df):
		# save df to fearther (shall be more efficient than pickle
		print("Saving results")
		print(f"to {self.model_saving_dir}")
		file_name = self.model_saving_dir+"/eval_results_df.p"
		# feather.write_feather(df, file_name)
		df.to_pickle(file_name)

	def track_state_actions(self, global_step, epoch, batch_idx, dm_rec_oneVrp, rewrite_rec_oneVrp, dm_id, eval_flag, or_tools_flag, or_tools_route=None, or_tools_cost=None):

		if eval_flag:
			if or_tools_flag:
				log_name = self.log_name_part + f"val_{dm_id}.txt"
				f = open(log_name, 'a')
				wr = csv.writer(f)
				wr.writerow(["or-tools solution"])
				wr.writerows(or_tools_route)
				wr.writerow([f"or-tools team avg cost {or_tools_cost}"])
				f.close()
			else:
				log_name = self.log_name_part+f"val_{dm_id}.txt"
				f = open(log_name, 'a')
				wr = csv.writer(f)
				wr.writerow([f"global step {global_step}",f"epoch {epoch}", f"batch_idx {batch_idx}"])
				for vrp_idx, vrp in enumerate(dm_rec_oneVrp):
					wr.writerow([f"state {vrp_idx}"])
					wr.writerows(vrp.vehicle_states)
					wr.writerow([f"team avg cost {vrp.team_avg_cost}"])
					if vrp_idx < len(rewrite_rec_oneVrp):
						wr.writerow(["chosen action"])  # index 2
						wr.writerows([rewrite_rec_oneVrp[vrp_idx][2][i] for i in range(len(rewrite_rec_oneVrp[vrp_idx][2]))])   # global action; rewrite rec in val case only contains global actions
						wr.writerow(["candidate actions"])  # index 0
						wr.writerows([rewrite_rec_oneVrp[vrp_idx][0][i] for i in range(len(rewrite_rec_oneVrp[vrp_idx][0]))])
						wr.writerow(["pred Q-values"])
						wr.writerow([rewrite_rec_oneVrp[vrp_idx][1]])
				f.close()
		else:
			# track training rewrite recs of each first vrp in batch to understand advantage each 40 steps
			log_name = self.log_name_part + f"train_{dm_id}.txt"
			f = open(log_name, 'a')
			wr = csv.writer(f)
			wr.writerow([f"global step {global_step}", f"epoch {epoch}", f"batch_idx {batch_idx}"])
			for vrp_idx, vrp in enumerate(dm_rec_oneVrp):
				wr.writerow([f"state {vrp_idx}"])
				wr.writerows(vrp.vehicle_states)
				wr.writerow([f"team avg cost {vrp.team_avg_cost}"])
				if vrp_idx < len(rewrite_rec_oneVrp):
					wr.writerows(rewrite_rec_oneVrp[vrp_idx][0])  # global action
			f.close()



"""
def write_summary(self, summary):
	print("global-step: %(global_step)d, avg-reward: %(avg_reward).3f" % summary)
	self.records.append(summary)
	df = pd.DataFrame(self.records)
	df.to_csv(self.log_name, index=False)
	self.best_reward = max(self.best_reward, summary['avg_reward'])
"""
