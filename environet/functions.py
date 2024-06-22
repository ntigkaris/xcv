"""Ntigkaris Alexandros"""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import sklearn
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import logging

from model import AQY_DS,AQY_NN
from constants import *

message = " Different versions are used, results may differ!"
try:
    assert np.__version__ == "1.21.6"
    assert pd.__version__ == "1.3.5"
    assert sklearn.__version__ == "1.0.2"
    assert torch.__version__ == "1.13.0+cu116"
except:
    logging.warning(message)

# functions

def get_score(
              y1:np.array,
              y2:np.array,
              ) -> tuple:

    return (
            mean_absolute_error(y1,y2),
            mean_squared_error(y1,y2,squared=False),
            r2_score(y1,y2),
            )
    
def IoA(
        y1:np.array,
        y2:np.array,
        ) -> float:
    """Index of Agreement"""
    return 1 - ( np.sum((y1-y2)**2) )/( np.sum((np.abs(y2-np.mean(y1))+np.abs(y1-np.mean(y1)))**2) )

def make_preprocessing() -> tuple:

    data = pd.read_csv(INDIR+"data.csv",)
    data["pm10_lag1"] = data.pm10.shift(1) # lag1 feature (pm10)
    data = data[1:].reset_index(drop=True)
    data.interpolate(inplace=True) # handle missing values

    holdout_data = pd.read_csv(INDIR+"holdout.csv")
    holdout_data["pm10_lag1"] = holdout_data.pm10.shift(1)
    holdout_data = holdout_data[1:].reset_index(drop=True)
    holdout_data.interpolate(inplace=True)

    scalerx = MinMaxScaler(feature_range=(0,1),)
    holdout_data[FEATURES] = scalerx.fit_transform(holdout_data[FEATURES])
    scalery = MinMaxScaler(feature_range=(0,1),)
    holdout_data[TARGET] = scalery.fit_transform(holdout_data[TARGET])

    # cross validation
    Fold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for n, (train_index, val_index) in enumerate(Fold.split(data[FEATURES], data[TARGET])):
        data.loc[val_index, "fold"] = int(n)
    data["fold"] = data["fold"].astype(int)

    return data,holdout_data,scalery

def train_fn(
             loader,
             model,
             criterion,
             optimizer,
             ) -> None:
    
    model.train()

    for X,y in loader:

        X = X.float()
        y = y.float()

        ypred = model(X)
        loss = criterion(ypred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    pass

def eval_fn(
            loader,
            model,
            criterion,
            ) -> np.array:

    model.eval()
    preds = []

    for X,y in loader:

        X = X.float()
        y = y.float()

        with torch.no_grad():
            ypred = model(X)
            loss = criterion(ypred,y)

        preds.append(ypred.numpy())

    preds = np.concatenate(preds)
    
    return preds

def make_NeuralNetwork(
                       df,
                       fold,
                       ) -> tuple:

    train_folds = df[df["fold"] != fold].reset_index(drop=True)
    eval_fold = df[df["fold"] == fold].reset_index(drop=True)

    scaler_x = MinMaxScaler(feature_range=(0,1),)
    train_folds[FEATURES] = scaler_x.fit_transform(train_folds[FEATURES])
        
    scaler_y = MinMaxScaler(feature_range=(0,1),)
    train_folds[TARGET] = scaler_y.fit_transform(train_folds[TARGET])

    eval_fold[FEATURES] = scaler_x.transform(eval_fold[FEATURES])
    eval_fold[TARGET] = scaler_y.transform(eval_fold[TARGET])

    train_ds = AQY_DS(train_folds)
    eval_ds = AQY_DS(eval_fold)

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        )

    eval_dl = DataLoader(
        eval_ds,
        batch_size=BATCH_SIZE*2,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        )

    torch.manual_seed(SEED)
    model = AQY_NN(
                    INPUT_DIM,
                    OUTPUT_DIM,
                    HIDDEN_DIM,
                    )

    optimizer = torch.optim.SGD(
                                model.parameters(),
                                lr=LR,
                                momentum=MOMENTUM,
                                )

    criterion = nn.MSELoss(reduction="mean")

    labels = eval_fold[TARGET].values

    for _ in range(EPOCHS):
        train_fn(train_dl,model,criterion,optimizer,)
        preds = eval_fn(eval_dl,model,criterion,)

    score = get_score(
                       scaler_y.inverse_transform(labels.reshape(-1,OUTPUT_DIM)),
                       scaler_y.inverse_transform(preds.reshape(-1,OUTPUT_DIM)),
                      )

    torch.save({"model":model.state_dict(),
                "predictions":preds},
                OUTDIR + f"model_fold{fold}.pth")

    return score

def make_LinearRegression(
                          df,
                          fold,
                          ) -> tuple:

    train_folds = df[df["fold"] != fold].reset_index(drop=True)
    eval_fold = df[df["fold"] == fold].reset_index(drop=True)

    scalerx = MinMaxScaler(feature_range=(0,1),)
    train_folds[FEATURES] = scalerx.fit_transform(train_folds[FEATURES])
        
    scalery = MinMaxScaler(feature_range=(0,1),)
    train_folds[TARGET] = scalery.fit_transform(train_folds[TARGET])

    eval_fold[FEATURES] = scalerx.transform(eval_fold[FEATURES])
    eval_fold[TARGET] = scalery.transform(eval_fold[TARGET])
    
    model = LinearRegression()
    model.fit(train_folds[FEATURES], train_folds[TARGET])

    y_pred = model.predict(eval_fold[FEATURES])
    score =  get_score(
                       scalery.inverse_transform(eval_fold[TARGET]),
                       scalery.inverse_transform(y_pred),
                      )

    pickle.dump(model, open(OUTDIR+f"baseline_fold{fold}.pkl","wb"))

    return score

def predict_holdout(
                      df,
                      scaler,
                      ) -> np.array:

    hds = AQY_DS(df)

    hdl = DataLoader(
                    hds,
                    batch_size=BATCH_SIZE*2,
                    pin_memory=True,
                    drop_last=False,
                    )

    ho_score = np.full((N_FOLDS,2,3),fill_value=np.nan)
    ho_labels = df[TARGET].values

    for f in range(N_FOLDS):

        current_model = AQY_NN(
                                INPUT_DIM,
                                OUTPUT_DIM,
                                HIDDEN_DIM,
                                )

        try:
            current_model.load_state_dict(torch.load(OUTDIR + f"model_fold{f}.pth")["model"])
        except:
            raise RuntimeError("Neural network has not been fit yet!")

        ho_preds_nn = []

        for hX,hy in hdl:
            with torch.no_grad():
                ypred = current_model(hX.float())
            ho_preds_nn.append(ypred.numpy())
        ho_preds_nn = np.concatenate(ho_preds_nn)

        ho_score[f,0,:2] = get_score(scaler.inverse_transform(ho_labels.reshape(-1,OUTPUT_DIM)),
                                    scaler.inverse_transform(ho_preds_nn.reshape(-1,OUTPUT_DIM)))[:2]
        ho_score[f,0,2] = IoA(scaler.inverse_transform(ho_labels.reshape(-1,OUTPUT_DIM)),
                            scaler.inverse_transform(ho_preds_nn.reshape(-1,OUTPUT_DIM)))
        
        try:
            current_baseline = pickle.load(open(OUTDIR+f"baseline_fold{f}.pkl","rb"))
        except:
            raise RuntimeError("Linear regression has not been fit yet!")
            
        ho_preds_lr = current_baseline.predict(df[FEATURES])

        ho_score[f,1,:2] = get_score(scaler.inverse_transform(ho_labels.reshape(-1,1)),
                                    scaler.inverse_transform(ho_preds_lr.reshape(-1,1)))[:2]
        ho_score[f,1,2] = IoA(scaler.inverse_transform(ho_labels.reshape(-1,1)),
                            scaler.inverse_transform(ho_preds_lr.reshape(-1,1)))
        
    return ho_score