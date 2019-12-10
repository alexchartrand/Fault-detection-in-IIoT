from os import path

# Output folders
FIGURE_FOLDER = "D:\\OneDrive\\Documents\\ETS\\Memoire\\Rapport\\Figures"
FIGURE_CHAP1_FOLDER = path.join(FIGURE_FOLDER, "revue_litterature")
FIGURE_MODELISATION_FOLDER = path.join(FIGURE_FOLDER, "modelisation_environnement")
FIGURE_CLASSIFICATION_FOLDER = path.join(FIGURE_FOLDER, "classification")

# Data folders
DATA_FOLDER = "D:\\OneDrive\\Documents\\ETS\\Memoire\\Simulation models\\DCMotorFault\Data"
NO_FAULT_MOTOR_FOLDER = path.join(DATA_FOLDER, "no_fault_motor")
FAULT_MOTOR_FOLDER = path.join(DATA_FOLDER, "fault_motor")
SIMULATION_MOTOR_FOLDER = path.join(DATA_FOLDER, "simulation")

# DF save folders
DATA_SAVE_FOLDER = "D:\\OneDrive\\Documents\\ETS\\Memoire\\IoT\\data"

# Fault type
FAULT_TYPE = {0: "aucune faute",
              1: "problème de roulement",
              2: "erreur du capteur de vitesse",
              3: "erreur du capteur de courant",
              4: "erreur du capteur de tension",
              5: "court-circuit de l'alimentation"}

FAULT_TYPE_SHORT = {0: "aucune",
              1: "roulement",
              2: "capteur de vitesse",
              3: "capteur de courant",
              4: "capteur de tension",
              5: "court-circuit"}