# setup

INDIR = "./"
OUTDIR = "./"
FEATURES = ["pm10_lag1","temp","rh"]
TARGET = ["pm25"]
INPUT_DIM = len(FEATURES)
OUTPUT_DIM = len(TARGET)
HIDDEN_DIM = 5
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-1
MOMENTUM = 0.8
SCALE = True #no-op
SEED = 42
N_FOLDS = 5