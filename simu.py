import mujoco
import mujoco.viewer
import subprocess
import os

# Pfade
MODEL_PATH = r"C:\Users\leona\Desktop\FuruPendel_MuJoCo\Pendel_description\urdf\robot.xml"
MUJOCO_BIN = r"C:\MuJoCo\bin\simulate.exe"  # <-- passe an, falls anders

# 1. Prüfen: Datei da?
if not os.path.exists(MODEL_PATH):
    print(f"FEHLER: {MODEL_PATH} nicht gefunden!")
    exit()

# 2. simulate.exe mit Modell starten
print("Starte MuJoCo simulate.exe...")
subprocess.Popen([MUJOCO_BIN, MODEL_PATH])

# Python-Viewer
try:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            data.ctrl[0] = 30  # Beispiel: 30 deg/s
            mujoco.mj_step(model, data)
            viewer.sync()
except ValueError as e:
    print(f"Fehler beim Laden: {e}")
    print("Tipp: Aktiviere <compiler balanceinertia='true'/> in MJCF!")