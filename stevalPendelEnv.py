import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import EzPickle
import mujoco
import mujoco.viewer # FÜR RENDERING IM 'human'-MODUS
import numpy as np
import os
import time # Für eine kontrollierte Simulationsgeschwindigkeit

# Definieren des relativen Pfades zur MuJoCo XML-Datei
MODEL_XML_PATH = r"stevalPendel\steval_pendel.xml"

class StevalPendelEnv(gym.Env, EzPickle):
    """
    Gymnasium Environment für das Steval Rotary Inverted Pendulum.
    Konfiguriert für das Training des Aufschwingens (Swing-up).
    
    WICHTIGE KONVENTION: 0 rad = UNTEN (Hängend) | pi rad = OBEN (Aufrecht)
    
    Beobachtungsraum (3D): [cos(theta), sin(theta), theta_dot] des Pendels.
    Aktionsraum (1D): Ziel-Winkelgeschwindigkeit für das Rotary-Gelenk.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"], 
        "render_fps": 100, # Effektive Steuerfrequenz (10 * 0.001s timestep)
    }

    def __init__(self, render_mode=None):
        EzPickle.__init__(self, render_mode=render_mode)

        # --------- Simulationsparameter ---------
        self.frame_skip = 10
        xml_path = MODEL_XML_PATH
        self.model_path = os.path.abspath(xml_path)
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"XML-Datei nicht gefunden unter: {self.model_path}")

        # Modell laden (erfordert 'limited="false"' im XML für 'pendel')
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        
        try:
            self.pendel_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pendel")
            self.rotary_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rotary")
            self.rotary_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rotary")
            # Hinweis: Der 'limited="false"' Fix muss im XML sein
        except ValueError as e:
            raise ValueError(f"Fehler beim Finden der Gelenk- oder Aktuator-Namen. Originalfehler: {e}")

        # MjData erstellen
        self.data = mujoco.MjData(self.model)

        # ************************************************
        # 🎯 FIX FÜR STARTZUSTAND (0 rad = UNTEN)
        # ************************************************
        # Setze das Pendel initial auf die Hängeposition (unten: 0 rad)
        downward_angle = 0.0 
        self.data.qpos[self.pendel_joint_id] = downward_angle
        # Synchronisiere die Kinematik
        mujoco.mj_forward(self.model, self.data)
        # ************************************************

        # --------- Rendering-Initialisierung ---------
        self.viewer = None 
        self.render_mode = render_mode
        
        # --------- Aktions- und Beobachtungsräume ---------
        obs_low = np.array([-1.0, -1.0, -np.inf], dtype=np.float64)
        obs_high = np.array([1.0, 1.0, np.inf], dtype=np.float64)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float64)

        ctrl_range = self.model.actuator_ctrlrange[self.rotary_actuator_id]
        act_low = np.array([ctrl_range[0]], dtype=np.float64)
        act_high = np.array([ctrl_range[1]], dtype=np.float64)
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float64)


    def _get_obs(self):
        """Berechnet den 3D-Beobachtungsvektor: [cos(theta), sin(theta), theta_dot]."""
        theta = self.data.qpos[self.pendel_joint_id]
        theta_dot = self.data.qvel[self.pendel_joint_id]
        return np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float64)
    
    def _get_reward(self, obs, action_clipped):
        """ Belohnung: Maximiert das Aufrechtsein (cos(theta)=-1) und minimiert die Geschw. """
        
        cos_upright_theta = obs[0]
        angle_dot = obs[2]
        
        # ************************************************
        # 🎯 KORREKTUR DER BELOHNUNG: 
        # Target UP = pi rad, cos(pi) = -1. Belohnung max bei cos = -1.
        # Skalierung von [-1, 1] zu [0, 1] (mit Umkehrung):
        # Wenn cos = -1 (Oben), Reward = (1 - (-1)) / 2 = 1.0
        # Wenn cos = 1 (Unten), Reward = (1 - 1) / 2 = 0.0
        upright_reward = (1 - cos_upright_theta) / 2
        # ************************************************
        
        # Velocity Penalty: Bestraft hohe Winkelgeschwindigkeiten des Pendels
        velocity_penalty = 0.01 * (angle_dot**2)
        
        # Control Penalty: Bestraft hohe Aktuator-Werte (Energieeffizienz)
        control_penalty = 0.005 * (action_clipped[0]**2)
        
        return upright_reward - velocity_penalty - control_penalty

    def step(self, action):
        """Führt einen Simulationsschritt aus."""
        
        action_clipped = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[self.rotary_actuator_id] = action_clipped[0]

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        observation = self._get_obs()
        reward = self._get_reward(observation, action_clipped)
        
        # Keine vorzeitige Terminierung beim Swing-up
        terminated = False 
        truncated = False
        info = {}
        
        if self.render_mode == "human":
            self.render()

        return observation, float(reward), terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Setzt die Umgebung zurück, Start in der unteren Hängeposition (0 rad)."""
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        
        # Start in der Hängeposition (0 rad = UNTEN)
        downward_angle = 0.0
        noise = self.np_random.uniform(low=-0.1, high=0.1)
        init_pos_pendel = downward_angle + noise
        init_vel_pendel = self.np_random.uniform(low=-0.1, high=0.1)
        
        self.data.qpos[self.pendel_joint_id] = init_pos_pendel
        self.data.qvel[self.pendel_joint_id] = init_vel_pendel
        self.data.qpos[self.rotary_joint_id] = 0.0
        self.data.qvel[self.rotary_joint_id] = 0.0

        if self.viewer is not None:
            self.viewer.sync() 

        observation = self._get_obs()
        info = {}

        return observation, info

    def render(self):
        """Implementiert das interaktive Rendering mit mujoco.viewer."""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            
            # Synchronisiere den Viewer mit den aktuellen Simulationsdaten
            self.viewer.sync()
        
        elif self.render_mode == "rgb_array":
            # (Platzhalter für Offscreen-Rendering)
            height, width = 480, 640
            if not hasattr(self, '_offscreen_renderer'):
                self._offscreen_renderer = mujoco.Renderer(self.model, height, width)
            self._offscreen_renderer.update_scene(self.data)
            return self._offscreen_renderer.render()


    def close(self):
        """Schließt den Viewer und räumt Ressourcen auf."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if hasattr(self, '_offscreen_renderer'):
            self._offscreen_renderer.close()


## 🚀 Hauptblock zum Testen der Umgebung
if __name__ == "__main__":
    
    # 1. Umgebung im human-Modus initialisieren
    print("Erstelle Umgebung im 'human'-Modus...")
    env = StevalPendelEnv(render_mode="human")
    
    # Simulationszeit und FPS
    RUNTIME_SECONDS = 10
    STEPS_PER_SECOND = env.metadata["render_fps"]

    print(f"Starte Testlauf für {RUNTIME_SECONDS} Sekunden ({RUNTIME_SECONDS * STEPS_PER_SECOND} Schritte).")
    
    try:
        obs, info = env.reset(seed=42)
        # cos(theta) sollte jetzt nahe 1.0 sein, da 0 rad = UNTEN
        print(f"Start-Zustand (cos(θ)={obs[0]:.2f}, theta_dot={obs[2]:.2f})")
        
        start_time = time.time()
        
        for i in range(RUNTIME_SECONDS * STEPS_PER_SECOND):
            
            # Test-Aktion: Starkes Hin- und Herschwingen (Sinuswelle)
            freq = 0.5 
            amplitude = 10.0 
            action = np.array([amplitude * np.sin(2 * np.pi * freq * (i / STEPS_PER_SECOND))])
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if env.viewer is not None and not env.viewer.is_running():
                print("Viewer geschlossen.")
                break

            # Kontrollierte Verzögerung für Echtzeit-Visualisierung
            elapsed_time = time.time() - start_time
            expected_time = (i + 1) / STEPS_PER_SECOND
            sleep_duration = expected_time - elapsed_time
            if sleep_duration > 0:
                time.sleep(sleep_duration)

    except Exception as e:
        print(f"\nFATALER FEHLER: {e}")
    finally:
        env.close()
        print("\nTest beendet. Viewer geschlossen.")