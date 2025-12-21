import gymnasium as gym
import random
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
import time

# Process one tick per step
TICK = 1                # Each step() call represents 1 tick
SLEEP_TIME = 0.001       # Sleep time for visualization (in seconds)
MAX_CAR = 100
ARRIVAL = 0

class Car:
    def __init__(self, id):
        self.id = id
        self.size = round(random.uniform(3.0, 5.0), 2)
        self.year = random.randint(10, 25)
        self.damage = round(random.uniform(0.0, 1.0), 2)
        # Patience in ticks
        self.patience = 100

    def __repr__(self):
        return (f"Car(id={self.id}, size={self.size}, "
                f"year={self.year}, damage={self.damage}, pat={self.patience})")




class GarageEnv(gym.Env):
    def __init__(self, debug=True, seed=None, reward_weights=None):  #*added for hw
        """Initialize the car repair shop simulation environment. #*added by danny

        Args:
            debug (bool, optional): Flag to enable debug mode. Defaults to True.
            seed (int, optional): Random seed for reproducibility. Defaults to None.

        Attributes:
            max_waiting (int): Maximum number of cars allowed in the waiting area (capacity = 3).
            total_cars_created (int): Total number of cars created during simulation.
            car_id_counter (int): Counter for assigning unique IDs to cars.
            success_count (int): Number of successfully repaired cars.
            removed_count (int): Number of cars removed due to patience exhaustion.
            waiting_area (list): Queue of Car objects waiting for repair.
            car_patience (dict): Maps car IDs to their remaining patience ticks.
            repair_status (dict): Status of each repair station ('A', 'B', 'C'), stores (car, remaining_ticks, assigned_ticks).
            current_time (int): Current simulation time in ticks.
            arrival_timer (int): Ticks until the next batch of cars arrives.
            action_space (spaces.Discrete): Action space with max_waiting * 3 + 1 discrete actions.
            observation_space (spaces.Box): Continuous observation space normalized to [0, 1].

        Example Flow:
            Tick 0:
            - waiting_area: [Car(id=0, size=4.5, year=15, damage=0.3, pat=100)]
            - car_patience: {0: 100}
            - repair_status: {'A': None, 'B': None, 'C': None}
            - success_count: 0, removed_count: 0
            
            After action=1 (assign Car 0 to station A):
            - waiting_area: []
            - car_patience: {} (car removed from patience dict)
            - repair_status: {'A': (Car(id=0,...), 35, 35), 'B': None, 'C': None}
            - success_count: 0
            
            After 35 ticks of repair:
            - repair_status: {'A': None, 'B': None, 'C': None}
            - success_count: 1 (car successfully serviced)

        Note:
            - obs_dim = 15: Observation dimension = 4 * max_waiting + 3
            - Observation space normalized to [0, 1] as Box(float32)
            - Patience decreases each tick; when patience <= 0, car is removed with -50 reward penalty
        
        
        Gymnasium spaces (how they are used in this env)
        -----------------------------------------------
        Gymnasium environments expose two key attributes that formally describe the
        input/output interface for an RL agent:

        1) action_space
            - A `gymnasium.spaces.Space` object that defines what actions are valid.
            - Here we use `spaces.Discrete(n)`, meaning the agent chooses an integer
                action in {0, 1, ..., n-1}.
            - In this environment:
                * action = 0 means "no-op" (do nothing this tick).
                * actions 1..(max_waiting*3) encode "assign a waiting car to a station".
                The encoding is:
                - idx = (action - 1) // 3     -> which car index in the waiting queue
                - station_idx = (action - 1) % 3 -> which station (A/B/C)
                This compact encoding is a common Gym-style pattern for turning a
                multi-choice decision (pick car, pick station) into a single Discrete id.

        2) observation_space
            - A `gymnasium.spaces.Space` object that defines the shape/range/type of
                observations returned by `reset()` and `step()`.
            - Here we use `spaces.Box(low=..., high=..., dtype=np.float32)`, which
                represents a continuous vector observation with per-dimension bounds.
            - In this environment the observation is intended to be a fixed-length
                vector that summarizes the current state (waiting cars + station status),
                normalized to [0, 1]. That normalization is why `low` is all zeros and
                `high` is all ones.

        How Gymnasium uses these:
        - `reset()` must return an observation that is contained in `observation_space`.
        - `step(action)` is expected to receive an action contained in `action_space`,
            and returns (observation, reward, terminated, truncated, info).
        - Many RL libraries rely on `action_space` and `observation_space` to build
            neural network input/output layers and to validate shapes/types.
        
        Is Gymnasium only used for action/observation spaces?
        > Not necessarily. In addition to `spaces.Discrete` and `spaces.Box`, Gymnasium
        is typically also used via:
        - Inheriting from `gymnasium.Env` (via `super().__init__()`), which defines
            the standard RL API contract (`reset()`, `step()`, optional `render()`,
            `close()`) and enables compatibility with Gymnasium-based libraries.
        - Seeding utilities and RNG handling (e.g., `Env.seed(...)` / `np_random`)
            when the environment supports reproducible randomness.
        - Optional wrappers, monitoring, vectorization, and validation that rely on
            `action_space` and `observation_space` metadata.
        To determine exactly which other Gymnasium features are used in *this codebase*,
        check the rest of the environment for imports/usages such as:
        `gymnasium.Env`, `gymnasium.utils.seeding`, `reset()`, `step()`,
        `render()`, `metadata`, and wrapper-related integration.        
    
        """
        
        super().__init__()
        self.debug = debug

        # Simulation parameters
        self.max_waiting = 3
        self.total_cars_created = 0
        self.car_id_counter = 0
        self.success_count = 0
        self.removed_count = 0

        #*added for hw: configurable reward shaping
        default_rw = {
            "assign": 0.5,
            "complete": 100.0,
            "expire": -50.0,
            "invalid": -1.0,
            "wait_penalty": -0.1,
            "time_penalty": -1.0,  #*added for hw
        }
        self.reward_weights = {**default_rw, **(reward_weights or {})}

        # Waiting area and garage status
        self.waiting_area = []                          # List[Car]
        self.car_patience = {}                          # {car.id: patience}
        self.repair_status = {'A': None, 'B': None, 'C': None}  # {station: (car, remaining ticks, assigned ticks)}

        # Time tracking
        self.current_time = 0                           # Cumulative time in ticks
        self.arrival_timer = random.randint(0, ARRIVAL)      # Ticks remaining until next batch arrival

        # Action/observation space
        self.action_space = spaces.Discrete(self.max_waiting * 3 + 1)

        # TODO: Define observation space size and normalization
        ############################ Modify this section if needed ################################
        obs_dim = 4 * self.max_waiting + 3
        low = np.zeros(obs_dim, dtype=np.float32)
        high = np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        ###################################################################################

        if seed is not None:
            self.seed(seed)



    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.debug:
            print('\n--- Simulation reset ---')

        # Initialize internal state
        self.total_cars_created = 0
        self.car_id_counter = 0
        self.success_count = 0
        self.removed_count = 0
        self.waiting_area.clear()
        self.car_patience.clear()
        self.repair_status = {'A': None, 'B': None, 'C': None}
        self.current_time = 0
        self.arrival_timer = random.randint(0, ARRIVAL)
        return self.get_observation(), {}

    def step(self, action):
        reward = 0.0

        # TODO: Define reward structure
        ############################ Modify this section if needed ################################
        # 1) Action processing: Strengthen reward for assignments
        if action != 0:
            idx = (action - 1) // 3
            station_idx = (action - 1) % 3
            station = ['A', 'B', 'C'][station_idx]
            if idx < len(self.waiting_area):
                car = self.waiting_area[idx]
                # Penalty if station is already occupied
                if self.repair_status[station] is not None:
                    reward += self.reward_weights["invalid"]  #*added for hw
                    if self.debug:
                        print(
                            f"[tick {self.current_time}] \033[91m!! Invalid assign: station {station} occupied\033[0m")
                else:
                    repair_time = self.get_repair_time(car, station)
                    self.repair_status[station] = (car, repair_time, repair_time)
                    self.waiting_area.pop(idx)
                    del self.car_patience[car.id]
                    reward += self.reward_weights["assign"]  #*added for hw
                    if self.debug:
                        print(
                            f"[tick {self.current_time}] \033[96m-> Assigned {car} to {station} (t={repair_time} ticks) \033[0m")

        # 2) Decrease arrival timer and spawn batch
        self.arrival_timer -= 1
        if self.arrival_timer <= 0:
            if len(self.waiting_area) < self.max_waiting:
                self._spawn_batch()
            self.arrival_timer = random.randint(0, ARRIVAL)

        # 3) Decrease patience and handle expiration (as before)
        for cid in list(self.car_patience):
            self.car_patience[cid] -= 1
            if self.car_patience[cid] <= 0:
                self._remove_car(cid)
                reward += self.reward_weights["expire"]  #*added for hw
                if self.debug:
                    print(f"[tick {self.current_time}] \033[35m-- Car {cid} patience expired, removed\033[0m")

        # 4) Update repair station status (as before)
        for st, status in list(self.repair_status.items()):
            if status is not None:
                car, rem, assigned = status
                rem -= 1
                if rem <= 0:
                    reward += self.reward_weights["complete"]  #*added for hw
                    self._finish_repair(st)
                else:
                    self.repair_status[st] = (car, rem, assigned)

        #*added for hw: per-tick queue penalty
        reward += self.reward_weights["wait_penalty"] * len(self.waiting_area)
        reward += self.reward_weights["time_penalty"]  #*added for hw: global per-step cost

        # 5) Update time and return
        self.current_time += 1
        obs = self.get_observation()
        done = (
                self.total_cars_created >= MAX_CAR
                and not self.waiting_area
                and all(s is None for s in self.repair_status.values())
        )

        ###################################################################################

        info = {'serviced': self.success_count, 'removed': self.removed_count}
        return obs, reward, done, False, info

    def get_observation(self):
        vec = np.zeros(4 * self.max_waiting + 3, dtype=np.float32)
        # TODO: Define observation vector
        ############################ Modify this section if needed ################################
        #*added for hw: encode up to max_waiting cars
        for i in range(min(len(self.waiting_area), self.max_waiting)):
            car = self.waiting_area[i]
            base = i * 4
            size_norm = (car.size - 3.0) / 2.0
            year_norm = (car.year - 10.0) / 15.0
            damage_norm = car.damage
            patience_norm = float(self.car_patience.get(car.id, car.patience)) / 100.0

            vec[base + 0] = np.clip(size_norm, 0.0, 1.0)
            vec[base + 1] = np.clip(year_norm, 0.0, 1.0)
            vec[base + 2] = np.clip(damage_norm, 0.0, 1.0)
            vec[base + 3] = np.clip(patience_norm, 0.0, 1.0)

        #*added for hw: station occupancy flags (A,B,C)
        offset = 4 * self.max_waiting
        vec[offset + 0] = 1.0 if self.repair_status['A'] is not None else 0.0
        vec[offset + 1] = 1.0 if self.repair_status['B'] is not None else 0.0
        vec[offset + 2] = 1.0 if self.repair_status['C'] is not None else 0.0
        ###################################################################################
        return np.concatenate([vec]).astype(np.float32)

    def _spawn_batch(self):
        slots = self.max_waiting - len(self.waiting_area)
        if slots <= 0:
            return
        batch = random.randint(1, slots)
        batch = min(batch, MAX_CAR - self.total_cars_created)
        for _ in range(batch):
            car = Car(self.car_id_counter)
            self.waiting_area.append(car)
            self.car_patience[car.id] = car.patience
            self.total_cars_created += 1
            self.car_id_counter += 1
            if self.debug:
                print(f"[tick {self.current_time}] \033[93m+ Car arrived: {car} (waiting={len(self.waiting_area)}/{self.max_waiting})\033[0m")

    def _remove_car(self, cid):
        self.waiting_area = [c for c in self.waiting_area if c.id != cid]
        del self.car_patience[cid]
        self.removed_count += 1

    def _finish_repair(self, station):
        car, _, _ = self.repair_status[station]
        self.repair_status[station] = None
        self.success_count += 1
        if self.debug:
            print(f"[tick {self.current_time}] \033[92m<- Car {car.id} done at {station} (total serviced={self.success_count})\033[0m")

    def can_accept(self, car, station):
        # All stations accept all vehicles
        return True


    def get_repair_time(self, car, station):
        """
        Station A: Repair time 34~36 ticks (random), longer for higher damage, accepts all ages
        Station B: Repair time 15~37 ticks (random), 30% reduction for vehicles with year>=20
        Station C: Repair time 12~37 ticks (random), 50% reduction for vehicles with size<=4.0
        """
        alpha = 10  
        if station == "A":
            base = random.randint(34, 36)
        elif station == "B":
            base = random.randint(15, 37)
        else:  # 'C'
            base = random.randint(12, 37)

        damage_ticks = int(round(car.damage * alpha))
        noise = random.randint(-1, 1)
        t = base + damage_ticks + noise

        if station == "B" and car.year >= 20:
            t = int(t * 0.7)
        if station == "C" and car.size <= 4.0:
            t = int(t * 0.5)

        # Clamp to min/max ticks
        low, high = (
            (34, 36) if station == "A" else (15, 37) if station == "B" else (12, 37)
        )
        return max(low, min(high, t))

if __name__ == "__main__":
    # 1) Create environment and set seed

    total_steps = 0

    for i in range(100):
        env = GarageEnv(debug=False, seed=i)

        # 2) Reset environment
        obs, _ = env.reset()
        total_reward = 0.0

        # 3) Test loop
        max_steps = 100_000
        step = 0
        while True:
            # -----------------------------
            # Rule-based baseline policy:
            # When there's an empty repair station, immediately assign the first car in queue (idx=0)
            # 0 = no-op, 1=A, 2=B, 3=C (all with idx=0)
            # Station reservation order: A, B, C
            action = 0
            if env.waiting_area:
                for station_idx, st in enumerate(['A', 'B', 'C']):
                    if env.repair_status[st] is None:
                        # Assign waiting_area[0] vehicle to st
                        action = station_idx + 1  # (action-1)//3 == 0, (action-1)%3 == station_idx
                        break
            # -----------------------------

            obs, reward, done, _, info = env.step(action)
            total_reward += reward

            if done:
                print("\n" + '\033[48;5;208m' + '\033[1m' + '\033[38;5;0m', end="")
                print("=== Simulation Finished ===")
                print(f" Steps taken : {step}")
                print(f" Total reward: {total_reward:.2f}")
                print(f" Serviced    : {info['serviced']}")
                print(f" Removed     : {info['removed']}")
                print('\033[0m')
                break

            step += 1
            if env.debug:
                time.sleep(SLEEP_TIME)
        else:
            print(f"\nStopped after {max_steps} steps (not done).")

        total_steps += step + (50 * info['removed']) # Apply penalty

    print(f"Avg steps : {total_steps / 100}")