import os
import gymnasium as gym
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


class FurutaPendulumEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 500,
    }

    def __init__(self, **kwargs):
        # Physikalische Grenzen (aus XML + Realität)
        self._max_velocity_joint0 = 50.0   # Rotor max ~50 rad/s
        self._max_velocity_joint1 = 50.0   # Pendel max ~50 rad/s
        self._max_torque = 0.5             # ctrlrange aus XML

        utils.EzPickle.__init__(self)

        # === OPTION 2: Unterschiedliche Grenzen ===
        observation_space = Box(
            low=np.array([-1.0, -1.0, -np.inf]),
            high=np.array([1.0, 1.0, np.inf]),
            dtype=np.float32,
        )

        # Pfad zum XML-Modell
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "model",
            "furuta_pendulum.xml",
        )

        MujocoEnv.__init__(
            self,
            model_path,
            frame_skip=2,
            observation_space=observation_space,
            **kwargs,
        )

        # Speicher für vorherige Aktion
        self.prev_action = np.zeros(1, dtype=np.float32)

        # Aktionsraum (passt zu XML)
        self.action_space = Box(
            low=-self._max_torque,
            high=self._max_torque,
            shape=(1,),
            dtype=np.float32,
        )

    def step(self, action):
        # --- Simulation ---
        self.do_simulation(action, self.frame_skip)
        self.bound_velocities()

        # --- Observation ---
        obs = self._get_obs()

        # --- Reward ---
        reward = self.calculate_reward_swingup(obs, action)

        # --- Update prev_action ---
        self.prev_action = np.array(action, dtype=np.float32).copy()

        # --- Termination ---
        terminated = not np.isfinite(obs).all()
        truncated = False

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.reset_model()
        return obs, {}

    def reset_model(self):
                qpos = self.init_qpos.copy()
                qvel = self.init_qvel.copy()

                # === NEUER START: ZUFÄLLIG NAHE DER UNTEREN POSITION (±10° um π) ===
                # 10 Grad in Radians: 10 * pi / 180 ≈ 0.1745 rad
                max_deviation = 10.0 * np.pi / 180.0
                
                # Der Startwinkel ist zentriert um np.pi (untere Position)
                phi_start = self.np_random.uniform(
                    low=np.pi - max_deviation, 
                    high=np.pi + max_deviation
                )
                # Pendel-Winkel setzen
                qpos[1] = phi_start 

                # Kleiner Zufallskick für Exploration (beibehalten)
                qpos += self.np_random.uniform(-0.02, 0.02, size=self.model.nq)
                qvel += self.np_random.uniform(-0.1, 0.1, size=self.model.nv)

                # === HINZUGEFÜGT: theta und theta_dot auf 0 setzen ===
                # Dies überschreibt den Zufallskick für Gelenk 0 (Rotor)
                qpos[0] = 0.0 # Setzt theta (Rotor-Position) auf 0
                qvel[0] = 0.0 # Setzt theta_dot (Rotor-Geschwindigkeit) auf 0
                # =====================================================

                self.set_state(qpos, qvel)

                # Reset prev_action
                self.prev_action = np.zeros(1, dtype=np.float32)

                obs = self._get_obs()
                return obs

    def bound_velocities(self):
        """Verhindert unphysikalische Geschwindigkeiten"""
        self.data.qvel[0] = np.clip(
            self.data.qvel[0], -self._max_velocity_joint0, self._max_velocity_joint0
        )
        self.data.qvel[1] = np.clip(
            self.data.qvel[1], -self._max_velocity_joint1, self._max_velocity_joint1
        )

    def _get_obs(self):
            #theta = self.data.qpos[0] # Rotor
            phi   = self.data.qpos[1] # Pendel

            theta_dot = self.data.qvel[0]
            phi_dot   = self.data.qvel[1]

            # Normalisierte Geschwindigkeiten
            theta_dot_norm = np.clip(theta_dot / self._max_velocity_joint0, -1.0, 1.0)
            phi_dot_norm   = np.clip(phi_dot / self._max_velocity_joint1, -1.0, 1.0)

            obs = np.array([
                np.cos(phi),          # 0: Pendel Oben/Unten
                np.sin(phi),          # 1: Pendel Links/Rechts
                phi_dot,         # 2: Pendel Geschwindigkeit
                
                # === HINZUGEFÜGT ===
               #np.cos(theta),        # 3: Rotor Position (cos)
               #np.sin(theta),        # 4: Rotor Position (sin)
                #theta_dot_norm,       # 5: Rotor Geschwindigkeit
            ], dtype=np.float32)

            return obs

    def calculate_reward_swingup(self, obs: np.array, a: np.array):
            
            # Entpacke die Observation (jetzt 6D)
            cos_phi, sin_phi, phi_dot_norm = obs

            # 1. BELOHNE HÖHE (Hauptziel)
            # (cos_phi ist 1.0 oben, -1.0 unten)
            height_reward = cos_phi 

            torque = a[0]
            action_penalty = -0.01 * torque**2 # Reduziert von 0.05

            reward = height_reward + action_penalty 
            
            return float(reward)

    def viewer_setup(self):
        assert self.viewer is not None
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent * 1.8
        v.cam.elevation = -25
        v.cam.azimuth = 135