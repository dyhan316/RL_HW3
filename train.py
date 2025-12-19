
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from env.car_repair_shop import GarageEnv
import json
import argparse

def load_config(path="./config/example_config.json"):
    with open(path, "r") as f:
        return json.load(f)

def make_env(debug=False, level=1, seed=None):
    def _init():
        env = GarageEnv(debug=debug, seed=None)
        env = Monitor(env)
        return env
    return _init

def main():
    # TODO: train policy and save
    ######################## Modify the section below if needed ########################
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # in case of using algorithms from stable_baseline3, use the following wrapper:
    # cfg = load_config(config_path)
    # env = DummyVecEnv([make_env(debug=False, level=args.level,seed=cfg.get("seed"))])
    # env = VecMonitor(env)

    ###################################################################################


if __name__ == "__main__":
    main()