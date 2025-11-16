import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import EzPickle
import mujoco
import mujoco.viewer
import numpy as np
import os
import time

MODEL_XML_PATH = r"stevalPendel/robot.xml"

class StevalPendelEnv(gym.Env, EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self, render_mode=None):
        super().__init__()
        EzPickle.__init__(self, render_mode=render_mode)

        self.frame_skip = 10
        self.model = mujoco.MjModel.from_xml_path(os.path.abspath(MODEL_XML_PATH))
        self.data = mujoco.MjData(self.model)

        self.pendel_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pendel")
        
        self.rotary_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rotary_velocity_control")
        self.observation_space = spaces.Box(low=np.array([-1, -1, -np.inf]),
                                             high=np.array([1, 1, np.inf]), dtype=np.float64)
        # Action ist die Ziel-Winkelgeschwindigkeit (Target Velocity) in rad/s
        self.action_space = spaces.Box(low=np.array([-15.0]), high=np.array([15.0]), dtype=np.float64)

        self.render_mode = render_mode
        self.viewer = None

    def _get_obs(self):
        theta = self.data.qpos[self.pendel_joint_id]
        theta_dot = self.data.qvel[self.pendel_joint_id]
        return np.array([np.cos(theta), np.sin(theta), theta_dot])

    def _get_reward(self, obs, action):
        cos_theta = obs[0]
        theta_dot = obs[2]
        upright = (1 - cos_theta) / 2          # max=1 bei cos=-1 (oben)
        vel_penalty = 0.01 * theta_dot**2
        ctrl_penalty = 0.005 * action[0]**2    # Bestrafung für hohe Zielgeschwindigkeiten
        return upright - vel_penalty - ctrl_penalty

    def step(self, action):
        # Der Action-Wert wird an den Geschwindigkeits-Aktuator übergeben (Ziel-Geschwindigkeit)
        self.data.ctrl[self.rotary_actuator_id] = np.clip(action, -15, 15)[0]
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        obs = self._get_obs()
        reward = self._get_reward(obs, action)
        
        # Beenden, wenn Pendel zu weit zur Seite kippt (optional)
        # done = bool(abs(self.data.qpos[self.pendel_joint_id]) > np.pi / 2) # Beispiel
        
        if self.render_mode == "human":
            self.render()
        
        # done=False, truncated=False (hier für endlose Episoce)
        return obs, float(reward), False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # 1. Die Startposition des Pendels (1 Radian)
        start_angle = 1.0 
        
        # 2. HOLEN SIE SICH DIE ADRESSE DES PENDELGELENKS IN QPOS
        pendel_qpos_adr = self.model.jnt_qposadr[self.pendel_joint_id]

        # 3. WEISEN SIE DIE WINKELPOSITION DER KORREKTEN ADRESSE ZU
        self.data.qpos[pendel_qpos_adr] = start_angle
        
        self.data.qvel[:] = 0
        
        # Führen Sie die Vorwärtskinematik aus, um die Zustände zu aktualisieren
        mujoco.mj_forward(self.model, self.data) 

        return self._get_obs(), {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

# Test
if __name__ == "__main__":
    # Stelle sicher, dass 'robot.xml' die korrekte XML-Struktur enthält!
    env = StevalPendelEnv(render_mode="human")
    #obs, _ = env.reset()
    # Da qpos auf 1 gesetzt ist, ist cos(1 rad) ≈ 0.54
    #print(f"Reset cos(θ) = {obs[0]:.4f} → sollte ≈ 0.5403 (nicht ganz unten)") 

    for i in range(2000):
        # Hier wird eine Zielgeschwindigkeit von 0 rad/s vorgegeben
        obs, r, _, _, _ = env.step(np.array([0.0])) 
        if i % 200 == 0:
            print(f"{i/100:.2f}s: cos(θ)={obs[0]:.6f}  reward={r:.4f}")
        time.sleep(0.01)

    env.close()