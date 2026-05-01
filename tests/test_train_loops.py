"""Tests for training callbacks and train loop modules."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TinyBinaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class TinyRegressorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def make_classifier_loader(n: int = 16):
    x = torch.randn(n, 1, 64, 64)
    y = (torch.rand(n) > 0.5).float()
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)


def make_regressor_loader(n: int = 16):
    x = torch.randn(n, 1, 64, 64)
    y = torch.rand(n, 5) * 2 - 1
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)


def test_early_stopping_max_mode():
    from gravlensai.train.callbacks import EarlyStopping

    es = EarlyStopping(patience=2, min_delta=1e-3, mode='max')
    assert es(0.5) is False
    assert es(0.5001) is False
    assert es(0.5002) is True


def test_model_checkpoint_saves(tmp_path):
    from gravlensai.train.callbacks import ModelCheckpoint

    model = TinyBinaryModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ckpt_path = tmp_path / 'best.pt'

    cb = ModelCheckpoint(str(ckpt_path), mode='max')
    assert cb(0.7, model, opt, 1, {'auc': 0.7}) is True
    assert ckpt_path.exists()
    assert cb(0.6, model, opt, 2, {'auc': 0.6}) is False


def test_classifier_train_epoch_and_validate():
    from gravlensai.train.train_classifier import train_one_epoch, validate_one_epoch

    model = TinyBinaryModel()
    loader = make_classifier_loader()
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device('cpu')

    train_metrics = train_one_epoch(model, loader, opt, criterion, device)
    val_metrics = validate_one_epoch(model, loader, criterion, device)

    assert set(train_metrics.keys()) == {'loss', 'precision', 'recall', 'auc'}
    assert set(val_metrics.keys()) == {'loss', 'precision', 'recall', 'auc'}


def test_train_classifier_smoke(tmp_path):
    from gravlensai.train.train_classifier import train_classifier

    model = TinyBinaryModel()
    train_loader = make_classifier_loader(24)
    val_loader = make_classifier_loader(24)
    save_path = tmp_path / 'classifier_best.pt'

    history = train_classifier(
        model,
        train_loader,
        val_loader,
        device=torch.device('cpu'),
        epochs=2,
        patience=2,
        save_path=str(save_path),
    )

    assert 'train' in history and 'val' in history and 'best_auc' in history
    assert save_path.exists()


def test_regressor_train_epoch_and_validate():
    from gravlensai.train.train_regressor import train_one_epoch, validate_one_epoch

    model = TinyRegressorModel()
    loader = make_regressor_loader()
    criterion = nn.HuberLoss(delta=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device('cpu')

    train_loss, train_rmse = train_one_epoch(model, loader, opt, criterion, device)
    val_loss, val_rmse = validate_one_epoch(model, loader, criterion, device)

    assert isinstance(train_loss, float)
    assert isinstance(val_loss, float)
    assert train_rmse.shape == (5,)
    assert val_rmse.shape == (5,)


def test_train_regressor_smoke(tmp_path):
    from gravlensai.train.train_regressor import train_regressor

    model = TinyRegressorModel()
    train_loader = make_regressor_loader(24)
    val_loader = make_regressor_loader(24)
    save_path = tmp_path / 'regressor_best.pt'

    history = train_regressor(
        model,
        train_loader,
        val_loader,
        device=torch.device('cpu'),
        epochs=2,
        patience=2,
        save_path=str(save_path),
    )

    assert 'train_loss' in history and 'val_loss' in history and 'best_val_loss' in history
    assert save_path.exists()
