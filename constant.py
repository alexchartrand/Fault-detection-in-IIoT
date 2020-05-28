from os import path

# Root project folder
ROOT_DIR = "D:/OneDrive/Documents/ETS/Memoire/IoT/"

# Output folders
FIGURE_FOLDER = "D:/OneDrive/Documents/ETS/Memoire/Rapport/Figures"
FIGURE_CHAP1_FOLDER = path.join(FIGURE_FOLDER, "revue_litterature")
FIGURE_MODELISATION_FOLDER = path.join(FIGURE_FOLDER, "modelisation_environnement")
FIGURE_CLASSIFICATION_FOLDER = path.join(FIGURE_FOLDER, "classification")

# Simulation data
SIMULATION_MOTOR_FOLDER = "D:/OneDrive/Documents/ETS/Memoire/Simulation models/DCMotorFault/Data"

# DF save folders
#DATA_SAVE_FOLDER = path.normpath("D:\\OneDrive\\Documents\\ETS\\Memoire\\IoT\\data")
SAVED_MODEL_FOLDER = "saved_model"
SAVED_CURVE_FOLDER = "saved_curve"

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

# Use Time
USE_TIME = False

# Data Index
TIME_IDX = 0
CURRANT_IDX = 1
SPEED_IDX = 2
VOLTAGE_IDS = 3
COMMAND_IDX = 5

if not USE_TIME:
    CURRANT_IDX -= 1
    SPEED_IDX -= 1
    VOLTAGE_IDS -= 1
    COMMAND_IDX -= 1