"""
PPI Hotspot Prediction - 配置文件
基于图注意力网络和ESM-2预训练模型
"""
import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, 'data')
PDB_DIR = os.path.join(DATA_DIR, 'pdb_files')
FEATURES_DIR = os.path.join(DATA_DIR, 'features')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

for dir_path in [DATA_DIR, PDB_DIR, FEATURES_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

DATA_FILE = os.path.join(BASE_DIR, 'merged_train.xlsx')
TEST_DATA_FILE = os.path.join(BASE_DIR, 'merged_test.xlsx')

ESM2_MODEL = 'facebook/esm2_t33_650M_UR50D'
ESM2_DIM = 1280

PSSM_DIM = 20
DSSP_DIM = 0
HMM_DIM = 30
TRADITIONAL_DIM = 27

INPUT_DIM = ESM2_DIM + PSSM_DIM + DSSP_DIM + HMM_DIM + TRADITIONAL_DIM

MAP_CUTOFF = 10.0
DIST_NORM = 15.0

HIDDEN_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 3
DROPOUT = 0.5

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0
BATCH_SIZE = 1
NUM_EPOCHS = 200
PATIENCE = 15

NUM_CLASSES = 2
RANDOM_SEED = 42
N_FOLDS = 5

FOCAL_ALPHA = 0.7
FOCAL_GAMMA = 2.0

LABEL_SMOOTHING = 0.05
USE_LABEL_SMOOTHING = True

USE_WEIGHTED_SAMPLER = True
USE_CLASS_WEIGHTS = True
USE_SMOTE = False

POS_WEIGHT_RATIO = 20.0

NOISE_FACTOR = 0
ACCUMULATION_STEPS = 1

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_PROPERTIES = {
    'A': {'hydrophobicity': 1.8, 'charge': 0, 'mw': 89, 'volume': 88, 'polarity': 0},
    'C': {'hydrophobicity': 2.5, 'charge': 0, 'mw': 121, 'volume': 108, 'polarity': 0},
    'D': {'hydrophobicity': -3.5, 'charge': -1, 'mw': 133, 'volume': 111, 'polarity': 1},
    'E': {'hydrophobicity': -3.5, 'charge': -1, 'mw': 147, 'volume': 138, 'polarity': 1},
    'F': {'hydrophobicity': 2.8, 'charge': 0, 'mw': 165, 'volume': 189, 'polarity': 0},
    'G': {'hydrophobicity': -0.4, 'charge': 0, 'mw': 75, 'volume': 60, 'polarity': 0},
    'H': {'hydrophobicity': -3.2, 'charge': 0.5, 'mw': 155, 'volume': 153, 'polarity': 1},
    'I': {'hydrophobicity': 4.5, 'charge': 0, 'mw': 131, 'volume': 166, 'polarity': 0},
    'K': {'hydrophobicity': -3.9, 'charge': 1, 'mw': 146, 'volume': 168, 'polarity': 1},
    'L': {'hydrophobicity': 3.8, 'charge': 0, 'mw': 131, 'volume': 166, 'polarity': 0},
    'M': {'hydrophobicity': 1.9, 'charge': 0, 'mw': 149, 'volume': 162, 'polarity': 0},
    'N': {'hydrophobicity': -3.5, 'charge': 0, 'mw': 132, 'volume': 114, 'polarity': 1},
    'P': {'hydrophobicity': -1.6, 'charge': 0, 'mw': 115, 'volume': 112, 'polarity': 0},
    'Q': {'hydrophobicity': -3.5, 'charge': 0, 'mw': 146, 'volume': 143, 'polarity': 1},
    'R': {'hydrophobicity': -4.5, 'charge': 1, 'mw': 174, 'volume': 173, 'polarity': 1},
    'S': {'hydrophobicity': -0.8, 'charge': 0, 'mw': 105, 'volume': 89, 'polarity': 1},
    'T': {'hydrophobicity': -0.7, 'charge': 0, 'mw': 119, 'volume': 116, 'polarity': 1},
    'V': {'hydrophobicity': 4.2, 'charge': 0, 'mw': 117, 'volume': 140, 'polarity': 0},
    'W': {'hydrophobicity': -0.9, 'charge': 0, 'mw': 204, 'volume': 227, 'polarity': 0},
    'Y': {'hydrophobicity': -1.3, 'charge': 0, 'mw': 181, 'volume': 193, 'polarity': 1},
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
