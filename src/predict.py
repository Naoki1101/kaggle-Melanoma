import numpy as np
import torch
from torch.autograd import Variable

import factory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_test(run_name, df, fold_df, cfg):
    all_preds = np.zeros(len(df))
    for fold_, col in enumerate(fold_df.columns):
        preds = predict_fold(run_name, df, cfg, fold_)
        all_preds += preds.reshape(-1) * fold_df[col].max()

    all_preds = 1 / (1 + np.exp(-all_preds.reshape(-1)))

    np.save(f'../logs/{run_name}/raw_preds.npy', all_preds)

    return all_preds


def predict_fold(run_name, df, cfg, fold_num):
    test_loader = factory.get_dataloader(df, cfg=cfg.data.test)

    model = factory.get_model(cfg, is_train=False).to(device)

    model.load_state_dict(torch.load(f'../logs/{run_name}/weight_best_{fold_num}.pt'))

    preds = []

    model.eval()
    for images, feats in test_loader:
        images = images.to(device)
        feats = feats.to(device)

        logits = model(images.float(), feats.float())
        preds.append(logits.cpu().detach().numpy())

    return np.concatenate(preds)