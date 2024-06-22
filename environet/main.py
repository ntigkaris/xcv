"""Ntigkaris Alexandros"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from tqdm.auto import tqdm
from constants import *
from functions import make_preprocessing, predict_holdout
from functions import make_NeuralNetwork, make_LinearRegression

#######################
# get train & test data
#######################

df,holdout_df,ho_scaler = make_preprocessing()

total_scores = np.full((N_FOLDS,2,3),fill_value=np.nan) # (n_folds,n_models,n_metrics)

##################################
# train linear regression and ann
##################################

for f in tqdm(range(N_FOLDS)):
    total_scores[f,:,:] = [
                            make_NeuralNetwork(df,f),
                            make_LinearRegression(df,f),
                           ]
train_score = np.mean(total_scores,axis=0)

print(f"\n[ANN]\tmae: {train_score[0,0]:.4f}\trmse: {train_score[0,1]:.4f}\tr2: {train_score[0,2]:.4f}\
        \n[LNR]\tmae: {train_score[1,0]:.4f}\trmse: {train_score[1,1]:.4f}\tr2: {train_score[1,2]:.4f}")

#########################
# get holdout data score
#########################

ho_scores = predict_holdout(holdout_df,ho_scaler)
holdout_score = np.mean(ho_scores,axis=0)

print(f"\n[ANN]\tmae: {holdout_score[0,0]:.4f}\trmse: {holdout_score[0,1]:.4f}\tIoA: {holdout_score[0,2]:.4f}\
        \n[LNR]\tmae: {holdout_score[1,0]:.4f}\trmse: {holdout_score[1,1]:.4f}\tIoA: {holdout_score[1,2]:.4f}")

############
# visualize
############

colors = ["red","#7680b5","#44bb66","orange","brown","black"]
colors_aux = [color for color in colors for _ in (0, 1)]
metrics = ["MAE","RMSE",r"$R^2$"]

plt.figure(figsize=(18,5))

for i in range(3):
    plt.subplot(1,3,i+1)
    plt.bar(
            range(12),
            np.concatenate([total_scores[:,:,i].flatten(),train_score[:,i].flatten()],axis=0),
            color=colors_aux,
            )
    plt.title(metrics[i])
    plt.xticks([])

plt.savefig(OUTDIR+"training.png",dpi=300)

plt.figure(figsize=(15,5))

for i in range(3):
    plt.subplot(1,3,i+1)
    plt.bar(
            range(6),
            np.concatenate([ho_scores[:,0,i].flatten(),holdout_score[0,i].flatten()],axis=0),
            color=colors,
            )
    plt.title(metrics[i]) if (i != 2) else plt.title("IoA")
    plt.yticks(np.arange(0,13,1)) if (i != 2) else plt.yticks(np.arange(0,1,0.1))
    plt.xticks([])

plt.savefig(OUTDIR+"inference.png",dpi=300)
