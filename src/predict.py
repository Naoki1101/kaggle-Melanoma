import numpy as np
import torch
from torch.autograd import Variable

import factory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_test(run_name, df, fold_df, cfg):
    all_preds = np.zeros((len(df), len(fold_df.columns)))
    for fold_, col in enumerate(fold_df.columns):
        preds = predict_fold(run_name, df, cfg, fold_)
        all_preds[:, fold_] = preds.reshape(-1)

    all_preds = np.max(all_preds, axis=1)
    all_preds = 1 / (1 + np.exp(-all_preds.reshape(-1)))

    np.save(f'../logs/{run_name}/raw_preds.npy', all_preds)

    return all_preds


def predict_fold(run_name, df, cfg, fold_num):
    test_loader = factory.get_dataloader(df, cfg=cfg.data.test)

    test_preds = np.zeros((len(test_loader.dataset), 
                           cfg.model.n_classes * cfg.data.test.tta.iter_num))
    test_batch_size = test_loader.batch_size

    model = factory.get_model(cfg, is_train=False).to(device)

    model.load_state_dict(torch.load(f'../logs/{run_name}/weight_best_{fold_num}.pt'))

    model.eval()
    for t in range(cfg.data.test.tta.iter_num):
        with torch.no_grad():
            for i, (images, feats) in enumerate(test_loader):
                images = images.to(device)
                feats = feats.to(device)

                preds = model(images.float(), feats.float())
                test_preds[i * test_batch_size: (i + 1) * test_batch_size, t * cfg.model.n_classes: (t + 1) * cfg.model.n_classes] = logits.cpu().detach().numpy()

        test_preds_tta = np.max(test_preds, axis=1)

    return test_preds_tta