from stable_baselines3 import PPO  #*added for hw
from stable_baselines3.common.vec_env import DummyVecEnv
from env.car_repair_shop import GarageEnv
import time
import argparse
import json  #*added for hw

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="")
    parser.add_argument('--config', default="./config/example_config.json")  #*added for hw
    args = parser.parse_args()

    # TODO: load model and test 
    ######################## Modify the section below #################################
    with open(args.config, "r") as f:
        cfg = json.load(f)

    seed = cfg.get("seed")
    reward_weights = cfg.get("reward_weights")

    env = DummyVecEnv([lambda: GarageEnv(debug=True, seed=seed, reward_weights=reward_weights)])

    if not args.model:
        raise ValueError("--model path is required")

    model = PPO.load(args.model, env=env)

    obs = env.reset()
    total_reward = 0.0
    serviced = 0
    removed = 0
    step = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # info is a list (one per env) because DummyVecEnv; take first element
        serviced = info[0].get('serviced', serviced)
        removed = info[0].get('removed', removed)
        total_reward += float(reward[0])  #*added for hw: avoid ndarray-to-scalar deprecation
        step += 1

        if done[0]:
            break
    ###################################################################################
    # NOTE: do not modify below
    total_steps = step + (50 * removed) # Apply penalty
    print(f"Finished in {total_steps} steps")
    print(f"Total reward: {total_reward:.2f},  Serviced: {serviced}, Removed: {removed}")

if __name__ == "__main__":
    # NOTE: code should run with python (or python3) test.py --model ./model/best_model.{any type} 
    main()