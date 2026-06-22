import numpy as np
import mujoco
import mujoco.viewer
import time
from stable_baselines3 import PPO
from train import BipedStabilizationEnv

def main():
    model_path = "../urdf/mjcf/bl-biped.xml"
    model_file = "biped_stabilizer"  
    
    env = BipedStabilizationEnv(model_path)
    model = PPO.load(model_file)
    
    print("Запуск симуляции. Ctrl+C для выхода.")
    
    obs, _ = env.reset()
    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -20
        
        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            
            # Вывод отладочной информации
            com_pos = env.data.subtree_com[0]
            feet_center = (env.data.geom_xpos[env.foot_left_id] + 
                          env.data.geom_xpos[env.foot_right_id]) / 2.0
            error = np.linalg.norm(com_pos[:2] - feet_center[:2])
            
            viewer.sync()
            time.sleep(0.01)

if __name__ == "__main__":
    main()