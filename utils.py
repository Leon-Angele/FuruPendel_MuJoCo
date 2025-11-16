import mujoco
import os

# Ihre Pfade
urdf_path = 'Pendel_description/urdf/Pendel.urdf'
mjcf_path = 'Pendel_description/urdf/robot.xml'

try:
    print(f"Versuche Konvertierung der URDF: {urdf_path}")
    
    # Lädt die korrigierte URDF und kompiliert sie
    model = mujoco.MjModel.from_xml_path(urdf_path)
    
    if model is None:
        raise Exception("Kompilierung fehlgeschlagen.")

    print("Kompilierung erfolgreich!")

    # Speichert als MJCF
    mujoco.mj_saveLastXML(mjcf_path, model)
    
    print(f"MJCF-Modell erfolgreich gespeichert unter: {mjcf_path}")

except Exception as e:
    print(f"Ein Fehler ist aufgetreten: {e}")