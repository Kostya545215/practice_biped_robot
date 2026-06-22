import gymnasium as gym
import numpy as np
import mujoco
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

class BipedStabilizationEnv(gym.Env):
    def __init__(self, model_path="../urdf/mjcf/bl-biped.xml"):
        super().__init__()
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 8 joints, normalized actions
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
        
        # Observations: qpos[7:15] + qvel[6:14] + orientation + angvel + imu
        obs_dim = 8 + 8 + 3 + 3 + 3
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initial pose: straight legs
        self.init_pose = np.zeros(8)
        
        # Joint indices in qpos
        self.joint_qpos_start = 7
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # Start at standing height
        self.data.qpos[2] = 0.55  # Higher start to give time
        
        # Straight orientation
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Initial pose with small noise
        noise = np.random.uniform(-0.01, 0.01, 8)
        self.data.qpos[self.joint_qpos_start:self.joint_qpos_start+8] = self.init_pose + noise
        
        # Zero initial velocities
        self.data.qvel[:] = 0.0
        
        mujoco.mj_forward(self.model, self.data)
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        # Joint positions and velocities
        qpos = self.data.qpos[self.joint_qpos_start:self.joint_qpos_start+8].copy()
        qvel = self.data.qvel[6:14].copy()
        
        # Body orientation (quaternion -> euler)
        quat = self.data.qpos[3:7].copy()
        euler = self._quat_to_euler(quat)
        
        # Body angular velocity
        angvel = self.data.qvel[3:6].copy()
        
        # IMU acceleration
        imu = self.data.sensor("imu_acc").data.copy()
        
        return np.concatenate([qpos, qvel, euler, angvel, imu]).astype(np.float32)
    
    def _quat_to_euler(self, quat):
        w, x, y, z = quat
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
        pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
        return np.array([roll, pitch, yaw])
    
    def step(self, action):
        # Scale actions from [-1, 1] to joint limits
        scaled_action = np.clip(action, -1.0, 1.0) * 1.5
        
        # Apply position commands
        self.data.ctrl[:] = scaled_action
        
        # Step simulation
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_terminated()
        truncated = False
        
        return obs, reward, terminated, truncated, {}
    
    def _compute_reward(self):
        
        # 1. Alive bonus
        alive = 1.0
        
        # 2. Upright reward (standard exponential)
        euler = self._quat_to_euler(self.data.qpos[3:7])
        upright = np.exp(-10.0 * (euler[0]**2 + euler[1]**2))  # roll & pitch
        
        # 3. Height penalty (quadratic around target)
        target_height = 0.50  # Target standing height
        current_height = self.data.qpos[2]
        height_cost = (current_height - target_height)**2
        
        # 4. Joint velocity penalty (encourage smooth motion)
        joint_vel = np.sum(np.square(self.data.qvel[6:14]))
        
        # 5. Control cost (encourage small actions)
        control_cost = 0.1 * np.sum(np.square(self.data.ctrl))
        
        reward = alive + 2.0 * upright - 5.0 * height_cost - 0.01 * joint_vel - control_cost
        
        return float(reward)
    
    def _is_terminated(self):
        """Termination conditions from standard implementations"""
        
        height = self.data.qpos[2]
        euler = self._quat_to_euler(self.data.qpos[3:7])
        
        # Fall down
        if height < 0.35:  # Higher threshold to avoid knee-standing
            return True
        
        # Too tilted
        if abs(euler[0]) > 0.8 or abs(euler[1]) > 0.8:  # ~45 degrees
            return True
        
        # Episode timeout (10 seconds at dt=0.01)
        if self.data.time > 10.0:
            return True
        
        return False

def make_env(model_path):
    def _init():
        env = BipedStabilizationEnv(model_path)
        env = Monitor(env)
        return env
    return _init

def main():
    model_path = "../urdf/mjcf/bl-biped.xml"
    
    env = DummyVecEnv([make_env(model_path)])
    check_env(BipedStabilizationEnv(model_path))
    
    eval_env = DummyVecEnv([make_env(model_path)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model_standard/",
        log_path="./logs/results_standard/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    progress_callback = ProgressBarCallback()
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device="cpu",
        tensorboard_log="./logs/tensorboard_standard/"
    )
    
    total_timesteps = 1_000_000
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, progress_callback],
        progress_bar=True
    )
    
    model.save("biped_stabilizer")

if __name__ == "__main__":
    main()