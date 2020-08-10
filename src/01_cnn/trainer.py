import gc
import sys
import time
import logging
from fastprogress import master_bar, progress_bar
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('../src')
import factory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def mixup(images, targets, alpha):
    indices = torch.randperm(images.size(0))
    shuffled_images = images[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    images = images * lam + shuffled_images * (1 - lam)
    targets = targets * lam + shuffled_targets * (1 - lam)

    return images, targets


def train_epoch(model, train_loader, criterion, optimizer, mb, cfg):
    model.train()
    avg_loss = 0.

    for images, labels in progress_bar(train_loader, parent=mb):
        images = images.to(device)
        labels = labels.to(device)

        r = np.random.rand()
        if cfg.data.train.mixup and r < 0.5:
            images, labels = mixup(images, labels, 1.0)

        preds = model(images.float())

        loss = criterion(preds.view(labels.shape), labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() / len(train_loader)
    del images, labels; gc.collect()
    return model, avg_loss


def val_epoch(model, valid_loader, criterion, cfg):
    model.eval()
    valid_preds = np.zeros((len(valid_loader.dataset), 
                            cfg.model.n_classes * cfg.data.valid.tta.iter_num))

    valid_preds_tta = np.zeros((len(valid_preds), cfg.model.n_classes))

    avg_val_loss = 0.
    valid_batch_size = valid_loader.batch_size
    
    for t in range(cfg.data.valid.tta.iter_num):
        with torch.no_grad():
            for i, (images, labels) in enumerate(valid_loader):
                images = images.to(device)
                labels = labels.to(device)

                preds = model(images.float())

                loss = criterion(preds.view(labels.shape), labels.float())
                valid_preds[i * valid_batch_size: (i + 1) * valid_batch_size, t * cfg.model.n_classes: (t + 1) * cfg.model.n_classes] = preds.cpu().detach().numpy()
                avg_val_loss += loss.item() / (len(valid_loader) * cfg.data.valid.tta.iter_num)
    
    for i in range(cfg.model.n_classes):
        preds_col_idx = [i + cfg.model.n_classes * j for j in range(cfg.data.valid.tta.iter_num)]
        valid_preds_tta[:, i] = np.mean(valid_preds[:, preds_col_idx], axis=1).reshape(-1)

    valid_preds_tta = 1 / (1 + np.exp(-valid_preds_tta))

    return valid_preds_tta, avg_val_loss


def train_model(run_name, df, fold_df, cfg):
    oof = np.zeros(len(df))
    cv = 0

    for fold_, col in enumerate(fold_df.columns):
        print(f'\n========================== FOLD {fold_} ... ==========================\n')
        logging.debug(f'\n========================== FOLD {fold_} ... ==========================\n')

        trn_x, val_x = df[fold_df[col] == 0], df[fold_df[col] > 0]
        val_y = val_x.loc[:33126][cfg.common.target]

        val_org_idx = np.where(val_x.index <= 33126)[0]

        train_loader = factory.get_dataloader(trn_x, cfg.data.train)
        valid_loader = factory.get_dataloader(val_x, cfg.data.valid)

        model = factory.get_model(cfg).to(device)
        
        criterion = factory.get_loss(cfg)
        optimizer = factory.get_optim(cfg, model.parameters())
        scheduler = factory.get_scheduler(cfg, optimizer)

        best_epoch = -1
        best_val_score = -np.inf
        mb = master_bar(range(cfg.data.train.epochs))

        train_loss_list = []
        val_loss_list = []
        val_score_list = []

        for epoch in mb:
            start_time = time.time()

            model, avg_loss = train_epoch(model, train_loader, criterion, optimizer, mb, cfg)

            valid_preds, avg_val_loss = val_epoch(model, valid_loader, criterion, cfg)

            val_score = factory.get_metrics(cfg.common.metrics.name)(val_y, valid_preds[val_org_idx])

            train_loss_list.append(avg_loss)
            val_loss_list.append(avg_val_loss)
            val_score_list.append(val_score)

            if cfg.scheduler.name != 'ReduceLROnPlateau':
                scheduler.step()
            elif cfg.scheduler.name == 'ReduceLROnPlateau':
                scheduler.step(avg_val_loss)
            
            elapsed = time.time() - start_time
            mb.write(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f} val_score: {val_score:.4f} time: {elapsed:.0f}s')
            logging.debug(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f} val_score: {val_score:.4f} time: {elapsed:.0f}s')

            if val_score > best_val_score:
                best_epoch = epoch + 1
                best_val_score = val_score
                best_valid_preds = valid_preds
                if cfg.model.multi_gpu:
                    best_model = model.module.state_dict()
                else:
                    best_model = model.state_dict()

        oof[val_x.index] = best_valid_preds.reshape(-1)
        cv += best_val_score * fold_df[col].max()

        torch.save(best_model, f'../logs/{run_name}/weight_best_{fold_}.pt')
        save_png(run_name, cfg, train_loss_list, val_loss_list, val_score_list, fold_)

        print(f'\nEpoch {best_epoch} - val_score: {best_val_score:.4f}')
        logging.debug(f'\nEpoch {best_epoch} - val_score: {best_val_score:.4f}')

    print('\n\n===================================\n')
    print(f'CV: {cv:.6f}')
    logging.debug(f'\n\nCV: {cv:.6f}')
    print('\n===================================\n\n')

    result = {
        'cv': cv,
    }

    np.save(f'../logs/{run_name}/oof.npy', oof)
    
    return result


def save_png(run_name, cfg, train_loss_list, val_loss_list, val_score_list, fold_num):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    ax1.plot(range(len(train_loss_list)), train_loss_list, color='blue', linestyle='-', label='train_loss')
    ax1.plot(range(len(val_loss_list)), val_loss_list, color='green', linestyle='-', label='val_loss')
    ax1.legend()
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.set_title(f'Training and validation {cfg.loss.name}')
    ax1.grid()

    ax2.plot(range(len(val_score_list)), val_score_list, color='blue', linestyle='-', label='val_score')
    ax2.legend()
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('score')
    ax2.set_title('Training and validation score')
    ax2.grid()

    plt.savefig(f'../logs/{run_name}/learning_curve_{fold_num}.png')