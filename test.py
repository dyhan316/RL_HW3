from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.car_repair_shop import GarageEnv
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="")
    args = parser.parse_args()

    # TODO: load model and test 
    ######################## Modify the section below #################################
    total_reward = 0
    serviced = 0
    removed = 0
    step = 0
    ###################################################################################
    # NOTE: do not modify below
    total_steps = step + (50 * removed) # Apply penalty
    print(f"Finished in {total_steps} steps")
    print(f"Total reward: {total_reward:.2f},  Serviced: {serviced}, Removed: {removed}")

if __name__ == "__main__":
    # NOTE: code should run with python (or python3) test.py --model ./model/best_model.{any type} 
    main()