import sys
import numpy as np
import torch
from torch.autograd import Variable

sys.path.append('../src')
import factory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_test(run_name, df, fold_df, cfg):
    all_preds = np.zeros((len(df), cfg.model.n_classes * len(fold_df.columns)))
    result_preds = np.zeros((len(df), cfg.model.n_classes))

    for fold_, col in enumerate(fold_df.columns):
        preds = predict_fold(run_name, df, cfg, fold_)
        all_preds[:, fold_ * cfg.model.n_classes: (fold_ + 1) * cfg.model.n_classes] = preds

    for i in range(cfg.model.n_classes):
        preds_col_idx = [i + cfg.model.n_classes * j for j in range(len(fold_df.columns))]
        result_preds[:, i] = np.mean(all_preds[:, preds_col_idx], axis=1)

    np.save(f'../logs/{run_name}/raw_preds.npy', result_preds)

    return result_preds


def predict_fold(run_name, df, cfg, fold_num):
    test_loader = factory.get_dataloader(df, cfg=cfg.data.test)

    test_preds = np.zeros((len(test_loader.dataset), 
                           cfg.model.n_classes * cfg.data.test.tta.iter_num))

    test_preds_tta = np.zeros((len(test_preds), cfg.model.n_classes))

    test_batch_size = test_loader.batch_size

    model = factory.get_model(cfg, is_train=False).to(device)
    model.load_state_dict(torch.load(f'../logs/{run_name}/weight_best_{fold_num}.pt'))

    model.eval()
    for t in range(cfg.data.test.tta.iter_num):
        with torch.no_grad():
            for i, (images) in enumerate(test_loader):
                images = images.to(device)

                preds, logits = model(images.float())
                test_preds[i * test_batch_size: (i + 1) * test_batch_size, t * cfg.model.n_classes: (t + 1) * cfg.model.n_classes] = preds.cpu().detach().numpy()
    
    for i in range(cfg.model.n_classes):
        preds_col_idx = [i + cfg.model.n_classes * j for j in range(cfg.data.test.tta.iter_num)]
        test_preds_tta[:, i] = np.mean(test_preds[:, preds_col_idx], axis=1).reshape(-1)

    test_preds_tta = 1 / (1 + np.exp(-test_preds_tta))

    return test_preds_tta