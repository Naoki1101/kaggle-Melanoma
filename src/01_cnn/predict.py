import sys
import numpy as np
import torch
from torch.autograd import Variable

sys.path.append('../src')
import factory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_test(run_name, df, fold_df, cfg):
    all_preds = np.zeros((len(df), cfg.model.n_classes * len(fold_df.columns)))
    all_feats = np.zeros((len(df), 256 * len(fold_df.columns)))

    for fold_, col in enumerate(fold_df.columns):
        preds, feats = predict_fold(run_name, df, cfg, fold_)
        all_preds[:, fold_ * cfg.model.n_classes: (fold_ + 1) * cfg.model.n_classes] = preds
        all_feats[:, fold_ * 256: (fold_ + 1) * 256] = feats

    for i in range(cfg.model.n_classes):
        preds_col_idx = [i + cfg.model.n_classes * j for j in range(len(fold_df.columns))]
        all_preds[:, i] = np.mean(all_preds[:, preds_col_idx], axis=1)

        feats_col_idx = [i + 256 * j for j in range(len(fold_df.columns))]
        all_feats[:, i] = np.mean(all_feats[:, preds_col_idx], axis=1).reshape(-1)

    np.save(f'../logs/{run_name}/raw_preds.npy', all_preds)
    np.save(f'../logs/{run_name}/test_feats.npy', all_feats)

    return all_preds


def predict_fold(run_name, df, cfg, fold_num):
    test_loader = factory.get_dataloader(df, cfg=cfg.data.test)

    test_preds = np.zeros((len(test_loader.dataset), 
                           cfg.model.n_classes * cfg.data.test.tta.iter_num))
    test_feats = np.zeros((len(test_loader.dataset), 
                           256 * cfg.data.test.tta.iter_num))

    test_preds_tta = np.zeros((len(test_preds), cfg.model.n_classes))
    test_feats_tta = np.zeros((len(test_preds), 256))

    test_batch_size = test_loader.batch_size

    model = factory.get_model(cfg, is_train=False).to(device)
    model.load_state_dict(torch.load(f'../logs/{run_name}/weight_best_{fold_num}.pt'))

    model.eval()
    for t in range(cfg.data.test.tta.iter_num):
        with torch.no_grad():
            for i, (images, feats) in enumerate(test_loader):
                images = images.to(device)
                feats = feats.to(device)

                preds, logits = model(images.float(), feats.float())
                test_preds[i * test_batch_size: (i + 1) * test_batch_size, t * cfg.model.n_classes: (t + 1) * cfg.model.n_classes] = preds.cpu().detach().numpy()
                test_feats[i * test_batch_size: (i + 1) * test_batch_size, t * 256: (t + 1) * 256] = logits.cpu().detach().numpy()
    
    for i in range(cfg.model.n_classes):
        preds_col_idx = [i + cfg.model.n_classes * j for j in range(cfg.data.test.tta.iter_num)]
        test_preds_tta[:, i] = np.mean(test_preds[:, preds_col_idx], axis=1).reshape(-1)

        feats_col_idx = [i + 256 * j for j in range(cfg.data.test.tta.iter_num)]
        test_feats_tta[:, i] = np.mean(test_feats[:, feats_col_idx], axis=1).reshape(-1)

    test_preds_tta = 1 / (1 + np.exp(-test_preds_tta))

    return test_preds_tta, test_feats_tta