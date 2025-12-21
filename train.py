
from stable_baselines3 import PPO  #*added for hw
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from env.car_repair_shop import GarageEnv
import json
import argparse
import os

def load_config(path="./config/example_config.json"):
    with open(path, "r") as f:
        return json.load(f)

def make_env(debug=False, level=1, seed=None, reward_weights=None):  # level unused but kept for compatibility
    def _init():
        env = GarageEnv(debug=debug, seed=seed, reward_weights=reward_weights)  #*added for hw
        env = Monitor(env)
        return env
    return _init

def main():
    # TODO: train policy and save
    ######################## Modify the section below if needed ########################
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/example_config.json")  #*added for hw
    parser.add_argument("--total_timesteps", type=int, default=None)  #*added for hw override
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("seed")
    reward_weights = cfg.get("reward_weights")
    n_envs = cfg.get("n_envs", 1)  #*added for hw

    env_fns = [make_env(debug=False, level=1, seed=(seed + i if seed is not None else None), reward_weights=reward_weights) for i in range(n_envs)]  #*added for hw
    env = DummyVecEnv(env_fns)  #*added for hw
    env = VecMonitor(env)

    total_timesteps = args.total_timesteps or cfg.get("total_timesteps", 100000)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=cfg.get("learning_rate", 3e-4),
        n_steps=cfg.get("n_steps", 2048),
        batch_size=cfg.get("batch_size", 64),
        n_epochs=cfg.get("n_epochs", 10),
        gamma=cfg.get("gamma", 0.99),
        gae_lambda=cfg.get("gae_lambda", 0.95),
        clip_range=cfg.get("clip_range", 0.2),
        ent_coef=cfg.get("ent_coef", 0.0),
        verbose=cfg.get("verbose", 1),
        tensorboard_log=cfg.get("tensorboard_log"),
        seed=seed,
    )

    model.learn(total_timesteps=total_timesteps)

    save_path = cfg.get("save_path", "model/ppo")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)  # saves a .zip containing the PyTorch weights and config
    print(f"Model saved to {save_path}.zip")

    ###################################################################################


if __name__ == "__main__":
    main()

"""
cliprange 0.2
(DLC) MacBook-Pro:HW3_skeleton eunmi$ python test.py --model ./model/ppo.zip --config ./config/example_config.json
Finished in 7457 steps
Total reward: -3951.70,  Serviced: 13, Removed: 87

(DLC) MacBook-Pro:HW3_skeleton eunmi$ python test.py --model ./model/ppo.zip --config ./config/example_config.json
Finished in 3090 steps
Total reward: 3829.50,  Serviced: 73, Removed: 27

ciprange 0.01
(DLC) MacBook-Pro:HW3_skeleton eunmi$ python test.py --model ./model/ppo.zip --config ./config/example_config.json
Finished in 1941 steps
Total reward: 7073.40,  Serviced: 93, Removed: 7


cliprange 0.001
(DLC) MacBook-Pro:HW3_skeleton eunmi$ python test.py --model ./model/ppo.zip --config ./config/example_config.json
Finished in 1750 steps
Total reward: 7311.60,  Serviced: 93, Removed: 7

cliprange 0.01, lr 1e-4
Finished in 2413 steps
Total reward: 5771.80,  Serviced: 85, Removed: 15

cliprange 0.01, lr 1e-5
Finished in 1310 steps
Total reward: 8529.30,  Serviced: 100, Removed: 0

cliprange 0.01, lr 1e-6
Finished in 2885 steps
Total reward: 4590.70,  Serviced: 78, Removed: 22


cliprange 0.01, lr 1e-5, batch size 256
Finished in 1891 steps
Total reward: 6893.00,  Serviced: 90, Removed: 10

cliprange 0.01, lr 1e-5, batch size 256, n_epochs 100
Finished in 2706 steps
Total reward: 6506.40,  Serviced: 98, Removed: 2

cliprange 0.01, lr 1e-5, batch size 256, n_epochs 1
Finished in 3282 steps
Total reward: 3500.30,  Serviced: 71, Removed: 29

cliprange 0.01, lr 1e-5, batch size 64, n_epochs 1
Finished in 3105 steps
Total reward: 3905.00,  Serviced: 73, Removed: 27

cliprange 0.01, lr 1e-5, batch size 64, n_epochs 1, total_steps 100000
Finished in 2684 steps
Total reward: 6484.90,  Serviced: 97, Removed: 3


cliprange 0.05, lr 1e-5, batch size 64, n_epochs 1, total_steps 100000
Finished in 2684 steps
Total reward: 6484.90,  Serviced: 97, Removed: 3


cliprange 0.0001, lr 1e-5, batch size 64, n_epochs 1, total_steps 100000
Finished in 2292 steps
Total reward: 6023.30,  Serviced: 85, Removed: 15


cliprange 0.0001, lr 1e-5, batch size 64, n_epochs 1, total_steps 100000, n_envs 100
Finished in 2292 steps
Total reward: 6023.30,  Serviced: 85, Removed: 15

// {
//   "seed": 123,
//   "learning_rate": 1e-5,
//   "n_steps": 100,
//   "batch_size": 64,
//   "n_epochs": 1,
//   "n_envs": 100,
//   "gamma": 0.99,
//   "gae_lambda": 0.95,
//   "clip_range": 0.0001,
//   "ent_coef": 0.0,
//   "verbose": 1,
//   "tensorboard_log": "./tensorboard/ppo_garage/lr1e-5_clip0.0001_n_epochs1_nevn100_tot_step100k/",
//   "total_timesteps": 100000,
//   "save_path": "model/ppo",
//   "reward_weights": {
//     "#": 0.5,
//     "complete": 100.0,
//     "expire": -50.0,
//     "invalid": -1.0,
//     "wait_penalty": -0.1
//   }
// }

#"""

"""
new ver defualt hting 
{
	"seed": 123,
	"learning_rate": 0.00001,
	"n_steps": 512,
	"batch_size": 256,
	"n_epochs": 10,
	"n_envs": 8,
	"gamma": 0.99,
	"gae_lambda": 0.95,
	"clip_range": 0.2,
	"ent_coef": 0.0,
	"verbose": 1,
	"tensorboard_log": "./tensorboard/ppo_garage/safe_defaults/",
	"total_timesteps": 500000,
	"save_path": "model/ppo",
	"reward_weights": {
		"#": 0.5,
		"complete": 100.0,
		"expire": -50.0,
		"invalid": -1.0,
		"wait_penalty": -0.1
	}
}
"""

"""NEW VER
#default
Finished in 8400 steps
Total reward: -5990.00,  Serviced: 0, Removed: 100
#* train loss decreaess well, but the performance is poor, and none were serviced.

#trying much larger learning rate, 0.001
Finished in 909 steps
Total reward: 9190.70,  Serviced: 100, Removed: 0

#trying even larger at 0.01
Finished in 946 steps
Total reward: 8899.90,  Serviced: 99, Removed: 1
#* could see policy update jumps (approx kl spike, policy_graidnet_loss spike early on) (maybe too large lr?)
#* also value loss showed overfitting after like 50k 

#tring 0.005 (in between)
Finished in 991 steps
Total reward: 9069.30,  Serviced: 99, Removed: 1

#trying smaller at 0.0001
#* stopped cuz too slow to learn

#trying 0.0005
Finished in 2659 steps
Total reward: 7231.50,  Serviced: 85, Removed: 15
#* also too slow

#===== Therefore lr will be set to 0.001 going forward =====
#* now from the 0.001 lr try, we say that the value loss was still not fully trained. therefore, total_timesteps
#* will be increased to 2 million
#trying 2 million steps (with lr 0.001 from now on) 
Finished in 934 steps
Total reward: 9646.50,  Serviced: 99, Removed: 1
#* reverting back to 500k steps since it didn't help

#*instead, let's play with the reward weights.
#*while the # of removed is 1 or zero (good), the total step is still too high (900+ )
#*therefore, will try to increase the wait penalty to encourage faster servicing

#trying wait penalty -0.5 (from -0.1)
Finished in 976 steps
Total reward: 8382.00,  Serviced: 98, Removed: 2
#* too strong, trying -0.3

#trying wait penalty -0.3
Finished in 1206 steps
Total reward: 8249.30,  Serviced: 94, Removed: 6
#* weird.. maybe it was penalized too much, leadning to actually less cars being serviced.
#*therfore going opposite direction, trying -0.05

#* weird.. trying -0.05
Finished in 1031 steps
Total reward: 9473.05,  Serviced: 97, Removed: 3
#*ok.. not good results. going back to -0.1

#trying wait penalty -0.1 (checking if we get what we expected previously)
Finished in 909 steps
Total reward: 9190.70,  Serviced: 100, Removed: 0
#* as expected


#*instead, maybe the car is being assigned to inefficient fixing garages. Therefore lowering the assign reward weight
#trying assign weight 0.1 (from 0.5)
Finished in 1017 steps
Total reward: 9296.60,  Serviced: 97, Removed: 3
#*too strong, maybe assign weight 0.3

#trying assign weight 0.3
Finished in 969 steps
Total reward: 9074.90,  Serviced: 100, Removed: 0
#*not the problem...

#*debug mode found that there are too many invalid assigns... Therefore, will increase the invalid penalty weight


#trying invalid weight -5.0 (from -1.0)
Finished in 963 steps
Total reward: 9643.50,  Serviced: 99, Removed: 1
#*maybe too harsh... trying -2.0

#trying invalid weight -2.0
Finished in 865 steps
Total reward: 9805.80,  Serviced: 100, Removed: 0
#* good! keeping this setting
#*now, we see that while all is serviced, the problem now is that the total steps is still high (865).
#* let's increase the weight penalty again, but with the invalid weight -2.0

#trying wait penalty -0.2 (from -0.1), invalid weight -2.0 (from -1.0)
Finished in 895 steps
Total reward: 9536.80,  Serviced: 100, Removed: 0
#*hmmm worse...? 
#*maybe try even larger wait penalty (maybe it's just stochasitiy, therefor trying larger wait penalty )

#trying wait penalty -0.3 (from -0.1), invalid weight -2.0 (from -1.0)
Finished in 1266 steps
Total reward: 8108.00,  Serviced: 92, Removed: 8
#* too strong, some where removed.. that's weird why is wait penalty causing removals...
#*maybe cuz it's doing suboptimal assignments to rush things, leading to more invalids and removals
#*therefore, trying the opposite, smaller wait penalty of -0.05

#trying wait penalty -0.05 (from -0.1), invalid weight -2.0 (from -1.0)
Finished in 1036 steps
Total reward: 9613.55,  Serviced: 98, Removed: 2

#let's subtract reward for every step to encourage faster servicing

#trying wait penality -0.1 (same), time penalty -0.5, invalid weight -2.0 (from -1.0)
Finished in 882 steps
Total reward: 9364.00,  Serviced: 100, Removed: 0
#*not good, maybe because time penalty is not large enough to matter? trhing -3.0

#trying wait penality -0.1 (same), time penalty -3.0, invalid weight -2.0 (from -1.0)
Finished in 927 steps
Total reward: 7009.60,  Serviced: 100, Removed: 0
#* ok not good.. moving time penality to -0.1

#trying wait penality -0.1 (same), time penalty -0.1, invalid weight -2.0 (from -1.0)
Finished in 919 steps
Total reward: 9567.30,  Serviced: 99, Removed: 1

#you know what? let's just go time penalty of 10 
#trying wait penality -0.1 (same), time penalty -10.0, invalid weight -2.0 (from -1.0)
Finished in 869 steps
Total reward: 1113.00,  Serviced: 100, Removed: 0



"""