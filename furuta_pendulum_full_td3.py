import gymnasium
from gymnasium.envs.registration import register
# Importieren Sie torch, wenn Sie die activation_fn in policy_kwargs verwenden möchten
# import torch 

from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import torch 
# --- 1. UMGEBUNGS-SETUP ---
max_episode_steps = 500

register(
    id="FurutaPendulum-v0",
    entry_point="furuta_pendulum_full:FurutaPendulumEnv",
    max_episode_steps=max_episode_steps,
)

# Schalter für Training/Evaluation
train_model = True # Auf True setzen, um das Training zu starten

# --- 2. HYPERPARAMETER ---

# Action Space Dimension für Furuta ist 1 (Drehmoment)
ACTION_DIM = 1
# Definiert den Action Noise für Exploration
action_noise = NormalActionNoise(mean=np.zeros(ACTION_DIM), sigma=0.1 * np.ones(ACTION_DIM))

# Erweiterte TD3-Algorithmus-Parameter
td3_params = dict(
    # Core RL Parameters
    learning_rate=5e-4,                      # Lernrate für Actor und Critic
    buffer_size=100_000_000,                   # Größe des Replay-Puffers
    learning_starts=10000,                    # NEU: Mehr Schritte vor Trainingsstart für bessere Initialdaten
    batch_size=256,                          # Größe der Mini-Batches
    tau=0.005,                               # Soft Target Update Rate (0.005 ist Standard)
    gamma=0.98,                              # Diskontfaktor

    # Training Frequenz & Effizienz
    train_freq=(1, "step"),                  # NEU: Sammle 1 Schritt, dann Update
    gradient_steps=1,                        # NEU: Führe 1 Gradientenschritt pro gesammeltem Schritt aus
    
    # TD3-spezifische Parameter
    policy_delay=2,                          # Policy (Actor) Update erfolgt alle 2 Critic-Updates
    target_policy_noise=0.2,                 # Standardabweichung des Target Policy Smoothing Noise
    target_noise_clip=0.5,                   # Clip-Wert für Target Policy Noise
    
    # Sonstige
    action_noise=action_noise,               # Der definierte Action Noise
    optimize_memory_usage=False,             # Speicheroptimierung 
)

# Policy-Hyperparameter (Netzwerkarchitektur)
policy_kwargs = dict(
    net_arch=[512, 512], # List of hidden layer sizes for the *shared* network
    activation_fn=torch.nn.ReLU,
)

# --- 3. TRAINING ODER EVALUATION ---

if train_model:
    # Parallele Umgebungen (empfohlen für schnellere Datensammlung)
    env = make_vec_env("FurutaPendulum-v0", n_envs=256)
    
    # Initialisierung des Modells mit entpackten Hyperparametern
    model = TD3(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="log",
        policy_kwargs=policy_kwargs,
        **td3_params 
    )

    print("Starte TD3 Training...")
    model.learn(total_timesteps=50_000_000, progress_bar=True)
    model.save("furuta_pendulum_rl/trained_agents/furuta_pendulum_full")
    
else:
    # Evaluation
    # Erstellt EINE Umgebung im Render-Modus
    env = gymnasium.make("FurutaPendulum-v0", render_mode="human")
    model = TD3.load("furuta_pendulum_rl/trained_agents/furuta_pendulum_full")

    # Gymnasium reset() gibt (obs, info) zurück. 
    # Wir rufen es auf, um den Wrapper zu initialisieren.
    env.reset() 

    # FEHLER BEHOBEN: Greife über .unwrapped auf die benutzerdefinierte Methode zu.
    obs = env.unwrapped.reset_model() 
    
    i = 0
    while True:
        action, _states = model.predict(obs, deterministic=True) # deterministic=True für Evaluation
        
        # Gymnasium step() gibt (obs, reward, terminated, truncated, info) zurück
        obs, rewards, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        env.render()
        i += 1
        
        # Stoppe, wenn Episode beendet oder maximaler Schritt erreicht
        if done:
            print(f"Episode beendet nach {i} Schritten.")
            break
            
    env.close()