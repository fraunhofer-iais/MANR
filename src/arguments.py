import argparse


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', type=bool, default=False)  # only implemented in eval
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true')  # creates default value of False if not present; call script like python .py --eval
    parser.add_argument('--load_model', type=str, default=None)  # checkpoint to start training OR to evaluate!
    parser.add_argument('--resume', type=int, default=0)# 4700 #0)  # load pretrained model; model name set in create_model
    parser.add_argument('--model_dir', type=str, default='../logs/NewData_SameTops_2agents10n_36afd_00000_30epochs_Z5_rewr30_gradClip0.05_eps0.15_alpha1e-06_lr0.0005_dstep200_drate0.9_maxStepPoolFill3_penalVal10.0')
    parser.add_argument('--advantage_file_name', type=str, default='advantage.txt')
    data_group = parser.add_argument_group('data')
    data_group.add_argument('--experiment_name', type=str, default="debug_pycharm")
    data_group.add_argument('--num_MLP_layers', type=int,
                            default=1)  # if you type 2, it will generate 3 hidden layers
    data_group.add_argument('--MLP_hidden_size', type=int, default=256)
    data_group.add_argument('--num_LSTM_layers', type=int, default=1)
    data_group.add_argument('--LSTM_hidden_size', type=int, default=256)  #256 # in all exps so far: 512,  512*2 = 1024 since bidirectional
    data_group.add_argument('--attention_size', type=int, default=16)
    data_group.add_argument('--embedding_size', type=int,
                            default=5)  # two possibilities: 3 (without xy) or 5 (with xy)
    # now: only nodeId, preNodeId and respective cost; before: default=7 (capacity information, ...)
    # second version with x-y-coordinates: then = 5; otherwise: 3
    data_group.add_argument('--embedding_size_vehicleState', type=int,
                            default=4)  # two possibilities: 2 (without xy) or 4 (with xy)
    # either only nodeId, preNodeId (cost unknown!)
    # second version with x-y-coordinates: then = 4, otherwise: 2
    train_group = parser.add_argument_group('train')
    # not really necessary; maybe in the future
    parser.add_argument('--train_proportion', type=float, default=1.0)
    parser.add_argument('--max_eval_size', type=int, default=1000)
    parser.add_argument('--processes', type=int, default=1)
    parser.add_argument('--param_init', type=float, default=0.1)
    # parser.add_argument('--seed', type=int, default=3112)
    parser.add_argument('--keep_last_n', type=int, default=5)   # models saved
    parser.add_argument('--log_interval', type=int, default=15)
    parser.add_argument('--log_name', type=str, default='model_0.csv')
    output_trace_group = parser.add_argument_group('output_trace_option')
    output_trace_group.add_argument('--output_trace_flag', type=str, default='nop',
                                    choices=['succeed', 'fail', 'complete', 'nop'])
    output_trace_group.add_argument('--output_trace_option', type=str, default='both', choices=['pred', 'both'])
    output_trace_group.add_argument('--output_trace_file', type=str, default=None)
    parser.add_argument('--heuristic', type=str, default="closest")  # closest, farthest

    # the ones currently fixed
    parser.add_argument('--allow_no_change_in_rule', type=bool, default=True)
    parser.add_argument('--take_x_y', type=bool, default=True)
    parser.add_argument('--rew_all_possible_embeddings', type=bool, default=True)

    train_group.add_argument('--dynamic_alpha', type=bool, default=False)
    train_group.add_argument('--max_reduce_steps', type=int,
                             default=3)  # 30 25, 10, 50, number of rewriting steps T_iter; for VRP: 200; default: 50
    train_group.add_argument('--dropout_rate', type=float, default=0.0)
    train_group.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'])


    # the ones with high potential for tuning
    # CURRENTLY COMMENTED OUT PENALTY SCALE THINGS
    # train_group.add_argument('--penalty_scale', type=float, default=0.5)  # new: factor of entropy penalty term
    # train_group.add_argument('--penalty_decay_steps', type=int, default=50)
    # train_group.add_argument('--penalty_decay_rate', type=float, default=0.9); "penalty_scale": tune.grid_search([0.1, 0.01, 0.001]),"penalty_decay_steps": tune.grid_search([50, 100]),
    train_group.add_argument('--gamma', type=float, default=0.5)  # decay factor in cumulative reward: (1, 0.5, 0.25, 0.125, 0.0625, 0.03125,0.016, 0.008, 0.004, 0.002, 0.001...) --> around 5 steps in future included
    train_group.add_argument('--lr_decay_steps', type=int, default=200)
    train_group.add_argument('--lr_decay_rate', type=float, default=0.9) # normally: 0.9
    train_group.add_argument('--epsilon_greedy', type=bool, default=True)

    # the ones usually tuned

    parser.add_argument('--Z', type=int, default=5)  # sample size Z 10
    train_group.add_argument('--num_epochs', type=int, default=30)#30 #22 #30 # 30
    train_group.add_argument('--epsilon', type=float, default=0.15) # typically: 0.05 #0.1) #0.01
    train_group.add_argument('--lr', type=float, default=0.0005)  #5e-4
    train_group.add_argument('--batch_size', type=int, default=32) # usually: 32   (CUDA: 8)
    train_group.add_argument('--value_loss_coef', type=float,
                             default=1e-5)  #0.01 influence of rule in loss --> very low default 0.01
    train_group.add_argument('--gradient_clip', type=float, default=0.05) #5.0
    train_group.add_argument('--max_steps_pool_filled', type=int, default=6)  # pool can be filled for maximally 3 consecutive steps; i.e. maximally allowed to decline two times. If declined for third them then penalty for this action
    train_group.add_argument('--penalty_val', type=float, default=10.0)


    parser.add_argument('--inf_num_runs', type=int, default=20)
    parser.add_argument('--path_init_route_nodes_eval', type=str, default="")
    parser.add_argument('--alternative_scenario', type=bool, default=False)  # no collab scenario in paper (solve TSP per agent with initially assigned nodes)



    #### mostly used from here
    parser.add_argument('--ray', type=bool, default=True) # TRUE usually
    parser.add_argument('--visdom', type=bool, default=True) # TRUE usually
    train_group.add_argument('--val_tracking', type=bool, default=True) # TRUE usually
    parser.add_argument('--eval_every_n', type=int, default=500) #500  #15  # n = batches; i.e. eval_every_n= 1 after each batch



   #parser.add_argument('--warm_start_model_file', type=str, default="/home/IAIS/npaul/comb_opt/pycharm_sync_marl_neural_rewriter/logs/paper_00_central_2agents10n_e742e_00000_Z5_rewr30_gradClip0.05_eps0.15_alpha1e-05_lr0.0005_dstep200_drate0.9_maxStepsPoolFilled3_penaltyVal10.0/ckpt-00004700")

    # initialize model parts with trained ones
    parser.add_argument('--load_partially', type=bool, default=False)

    #"/home/IAIS/npaul/comb_opt/pycharm_sync_marl_neural_rewriter/logs/paper_00_central_2agents20n_3c1aa_00000_Z10_rewr40_gradClip0.05_eps0.15_alpha1e-06_lr0.0005_dstep200_drate0.9_maxStepsPoolFilled3_penaltyVal10.0/ckpt-00004700"
    # OLD NewData 2ag 10 nodes:"/home/IAIS/npaul/comb_opt/pycharm_sync_marl_neural_rewriter/logs/NewData_SameTops_2agents10n_36afd_00000_30epochs_Z5_rewr30_gradClip0.05_eps0.15_alpha1e-06_lr0.0005_dstep200_drate0.9_maxStepPoolFill3_penalVal10.0/ckpt-00004700"
    # OLD NewData 2ag 20 nodes: "/home/IAIS/npaul/comb_opt/pycharm_sync_marl_neural_rewriter/logs/NewData_SameTops_2ag20n_1e543_00001_Z10_rewr40_gradClip0.05_eps0.15_alpha5e-07_lr0.0005_maxStepPoolFill3_penalVal10.0/ckpt-00004700"
    parser.add_argument('--load_partially_model_file', type=str, default="/home/IAIS/npaul/comb_opt/pycharm_sync_marl_neural_rewriter/logs/NewData_2ag10n_7377d_00001_DiffQLayers_PolicyWithContextInfo_Z5_rewr30_gradClip0.05_eps0.15_alpha1e-06_lr0.0005_maxStepPoolFill3_penalVal10.0/ckpt-00004700")
    parser.add_argument('--init_policy', type=bool, default=True)
    parser.add_argument('--init_decoders', type=bool, default=True)  # both state encoder with and without costs
    parser.add_argument('--init_Q_layers', type=bool, default=False)


    # warm start training: perturb init parameters loaded via load_partially
    parser.add_argument('--warm_start', type=bool, default=False)
    # lambda in the warm start paper to shrink parameters with
    parser.add_argument('--shrink_lambda', type=float, default=0.6) # <1
    parser.add_argument('--perturb_sigma', type=float, default=0.001) # standard deviation


    # make rule policy less determined (if too sure too difficult and unstable to learn)
    parser.add_argument('--rule_temperature_scaling', type=bool, default=False)  # if True: kept True for x batch updates (afterwards set to false; o/w policy can never get determined)

    parser.add_argument('--rule_temperature_adaptive', type=bool, default=True)  # adapt temperature over time (additively equally to become max 1; init rule temp and batch thresh
    parser.add_argument('--rule_temperature_scaling_batch_thresh', type=int, default=2)  # until then scaling applied, i.e. after thresh many model updates it's not applied anymore
    parser.add_argument('--rule_temperature', type=float, default=0.5)  # if temperature < 1, the probability distribution becomes more equally distributed


    # give less weight to local reordering actions in rule policy loss --> more emphasize on pool actions which shall be newly learned
    parser.add_argument('--weigh_local_reord_in_rule_loss', type=bool, default=False)

    parser.add_argument('--loss_weight_local_reord_act', type=float, default=0.8) # <<1 to ignore local reordering actions and automatically focus on pool interactions which have to be newly learned in a new agent setup
    parser.add_argument('--loss_weight_pool_act', type=float, default=1.2)

    # deepmind TL approach: expected entropy regularised distillation
    parser.add_argument('--tl_entropy_distill', type=bool, default=False)

    parser.add_argument('--tl_entropy_distill_batch_thresh', type=int, default=10000) #300 #5000; needs ot be very high to have no influence (eps-stopping criterion instead used)
    #### UPDATE: not weighted; but therefore PER Local reord action (before sum over all rewriting steps)
    parser.add_argument('--eps_weighted_tl', type=float, default = 3.0)#0.01) # if tl_weight * transfer loss < eps, then abschalten
    parser.add_argument('--tl_teacher_num_agents', type=int, default=2)  # number of agents in teacher model
    parser.add_argument('--tl_teacher_model_path', type=str,
                        default="/home/IAIS/npaul/comb_opt/pycharm_sync_marl_neural_rewriter/logs/NewData_2ag10n_7377d_00001_DiffQLayers_PolicyWithContextInfo_Z5_rewr30_gradClip0.05_eps0.15_alpha1e-06_lr0.0005_maxStepPoolFill3_penalVal10.0/ckpt-00004700")
    parser.add_argument('--tl_loss_weight', type=float, default=0.01)  #0.01

    # only update Q and encoders for the first 500 global steps
    parser.add_argument('--turn_off_alpha_temporary', type=bool, default=False)

    #10n_5a_100SameTops_Vel05EdgeVel05_5000s_train
    #10n_5a_30SameTops_Vel05EdgeVel05_600s_val
    #10n_5a_diffTops_Vel05EdgeVel05_5000s_train
    #10n_5a_diffTops_Vel05EdgeVel05_600s_val

    #
    #10n_5a_100SameTops_Vel05EdgeVel05_5000s_train
    #10n_5a_30SameTops_Vel05EdgeVel05_600s_val
    #10n_5a_OneTop_5600CostMatrices_train_val
    data_group.add_argument('--train_dataset', type=str,
                            default='/home/IAIS/npaul/comb_opt/pycharm_sync_marl_neural_rewriter/data/vrp/newAgVelAgEdgeVel/10n_2a_100SameTops_Vel05EdgeVel05_5000s_train.p')  # vrp_10_2_cl_ellipse_same_1000s, vrp_10_1_1000s_train, vrp_20_1_1000s_train, vrp_5_1_1000s_train,vrp_5_1_1000s_same_train, vrp_5_1_1000s_train ,vrp_20_5_100000s_train.p ,
    data_group.add_argument('--val_dataset', type=str,
                            default='/home/IAIS/npaul/comb_opt/pycharm_sync_marl_neural_rewriter/data/vrp/newAgVelAgEdgeVel/10n_2a_30SameTops_Vel05EdgeVel05_600s_val.p')  # vrp_10_1_1000s_val,vrp_5_1_1000s_val, vrp_5_1_1000s_same_val, vrp_5_1_1000s_val,vrp_20_5_100000s_val.p,
    data_group.add_argument('--test_dataset', type=str,
                            default='/home/IAIS/npaul/comb_opt/pycharm_sync_marl_neural_rewriter/data/vrp/newAgVelAgEdgeVel/10n_2a_30SameTops_Vel05EdgeVel05_600s_test.p')
    # vrp_10_1_1000s_test,vrp_5_1_1000s_test, vrp_5_1_1000s_same_test, vrp_5_1_1000s_test, ,vrp_20_5_100000s_test.p',


    return parser




# removed params

#parser.add_argument('--softmax_Q', type=bool, default=False)



# parser.add_argument('--input_format', type=str, default='DAG', choices=['seq', 'DAG'])
# parser.add_argument('--num_sample_rewrite_pos', type=int, default=10)  # T_w
# parser.add_argument('--num_sample_rewrite_op', type=int, default=10)  # T_u
# parser.add_argument('--cont_prob', type=float, default=0.5)  # init p_c
# train_group.add_argument('--only_rule_can_do_nothing', type=bool, default=True)
# parser.add_argument('--max_loss', type=bool, default=False)  # take sum