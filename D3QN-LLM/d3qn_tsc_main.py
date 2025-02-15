import torch.optim as optim
import tsc_env as env
import d3qn_agent as d3a


SEEDS = (12,23)
d3qn_results = []
best_agent, best_eval_score = None, float('-inf')

for seed in SEEDS:


    #'''
    if seed == 23:
        print("23")
        break
    #'''


    model_number = "_241106[eval]_snowy_period1.0_start70_0.5_TSC_Z%d_C%s_S%d_seed%d" % (int(0.8 * 100), int(0.2 * 100), 5, seed)
    eval_model = "/home/suzzang77/finalProject/Projects/tsc_real/d3qn_models/d3qn_241024[train]_start70_0.5_TSC_Z80_C20_S5_seed12.pt"
    message = ''

    environment_settings = {
        'env_name': 'TSC',
        'gamma': 0.99,
        'max_minutes': 1200,
        'max_episodes': 300,  
        'goal_mean_100_reward': 0  
    }

    value_model_fn = lambda nS, nA: d3a.FCDuelingQ(nS, nA, hidden_dims=(512, 128))
    value_optimizer_fn = lambda net, lr: optim.RMSprop(net.parameters(), lr=lr)
    value_optimizer_lr = 0.001  
    max_gradient_norm = float('inf')
    

    training_strategy_fn = lambda: d3a.EGreedyExpStrategy(init_epsilon=1.0,  
                                                      min_epsilon=0.3, 
                                                      decay_steps=20000)
    evaluation_strategy_fn = lambda: d3a.GreedyStrategy()

    replay_buffer_fn = lambda: d3a.ReplayBuffer(max_size=50000, batch_size=64)

    n_warmup_batches = 5
    update_target_every_steps = 1
    tau = 0.1

    env_name, gamma, max_minutes, max_episodes, goal_mean_100_reward = environment_settings.values()

    agent = d3a.DuelingDDQN(replay_buffer_fn,
                        value_model_fn,
                        value_optimizer_fn,
                        value_optimizer_lr,
                        max_gradient_norm,
                        training_strategy_fn,
                        evaluation_strategy_fn,
                        n_warmup_batches,
                        update_target_every_steps,
                        tau)

    #tsc = env.TSC(seed) #train 환경

    tsc_eval = env.eval_TSC(seed) #evaluate 환경

    # 학습
    #result, final_eval_score, training_time, wallclock_time = agent.train(
    #    tsc, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward, model_number)


    # 평가
    final_eval_score, score_std, reward = agent.su_evaluate(eval_model,tsc_eval,model_number,100)
    print("Reward:",final_eval_score)
    
'''
    d3qn_results.append(result)
    if final_eval_score > best_eval_score:
        best_eval_score = final_eval_score
        best_agent = agent
d3qn_results = np.array(d3qn_results)
'''
