# model_factory.py

from .path_helper import add_base_to_sys_path
add_base_to_sys_path(2)
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from tabpfn import TabPFNClassifier
from sklearn.neural_network import MLPClassifier
import global_config
from sklearn.svm import SVC,SVR
def set_params(
    task=None,
    model_name=None,
    random_state=None,
    task_name=None,
    metric=None,
    n_estimators=None,
    learning_rate=None,
    other_model=None,
    compute_tokens=False
):
    """Update global configuration parameters if provided.
    
    Only updates parameters when their values are not None.
    """
    # Update task-related parameters
    if task is not None:
        global_config.task = task
    
    # Update general model parameters
    if model_name is not None:
        global_config.model_name = model_name
    
    if random_state is not None:
        global_config.data_pre["random_state"] = random_state
    
    if task_name is not None:
        global_config.task_name = task_name
    
    if metric is not None:
        global_config.metric = metric
    
    # Update XGBoost-specific parameters
    if n_estimators is not None:
        global_config.xgboost["n_estimators"] = n_estimators
    
    if learning_rate is not None:
        global_config.xgboost["learning_rate"] = learning_rate
    if other_model != "" and other_model !=None:
        global_config.other_model=other_model
    else:
        global_config.other_model=""
    if compute_tokens:
        global_config.compute_tokens=compute_tokens

def get_model(model_name=None, task=None):
    # 动态获取当前配置值
    model_name = global_config.model_name if model_name is None else model_name
    task = global_config.task if task is None else task

    if model_name == "xgboost":
        if task == "regression":
            return XGBRegressor(
                n_estimators=global_config.xgboost["n_estimators"],
                # n_estimators=50,
                learning_rate=global_config.xgboost['learning_rate'],
                max_depth=6,
                random_state=42,
                tree_method='hist'
            )
        else:
            return XGBClassifier(
                n_estimators=global_config.xgboost["n_estimators"],
                learning_rate=global_config.xgboost['learning_rate'],
                max_depth=6,
                random_state=42,
                tree_method='hist'
            )
    elif model_name == "mlp":
        return TorchMLPClassifier(
                input_dim=None,
                hidden=128,
                n_blocks=3,
                dropout=0.2,
                lr=1e-3,
                weight_decay=1e-4,
                batch_size=64,
                max_epochs=50,
                patience=5,
                validation_fraction=0.2,
                scaler=True,
                verbose=1
            )
    elif model_name == "random_forest":
        return RandomForestRegressor(random_state=42) if task == "regression" else RandomForestClassifier(random_state=42)

    elif model_name == "linear":
        return TorchMLPClassifier(input_dim=None, hidden=128, n_blocks=3, batch_size=64, max_epochs=50) if task == "regression" else None


    elif model_name == "lightgbm":
        return LGBMRegressor(random_state=42) if task == "regression" else LGBMClassifier(               n_estimators=global_config.xgboost["n_estimators"],
                learning_rate=global_config.xgboost['learning_rate'],random_state=42)

    elif model_name == "catboost":
        return CatBoostRegressor(n_estimators=global_config.xgboost["n_estimators"],verbose=0, random_state=42) if task == "regression" else CatBoostClassifier(n_estimators=global_config.xgboost["n_estimators"],verbose=0, random_state=42)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

        
        
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
import time
import pandas as pd

# -------------------------
# Model components (unchanged)
# -------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = out + identity
        out = self.norm(out)
        return out

class MediumMLP(nn.Module):
    def __init__(self, input_dim=20, hidden=128, n_blocks=3, num_classes=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden)
        self.act = nn.SiLU()
        self.blocks = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.act(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

# -------------------------
# Sklearn-style wrapper with lazy model creation
# -------------------------
class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 input_dim=None,         # can be None -> inferred in fit
                 hidden=128,
                 n_blocks=3,
                 dropout=0.2,
                 lr=1e-3,
                 weight_decay=1e-4,
                 batch_size=64,
                 max_epochs=100,
                 patience=10,
                 validation_fraction=0.1,
                 device=None,
                 scaler=True,
                 seed=42,
                 verbose=1):
        self.input_dim = input_dim
        self.hidden = hidden
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = scaler
        self.seed = seed
        self.verbose = verbose

        # placeholders (will be set in fit)
        self.model_ = None
        self.scaler_ = None
        self.label_encoder_ = None
        self.classes_ = None
        self.history_ = {}

    def _set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if "cuda" in self.device:
            torch.cuda.manual_seed_all(self.seed)

    def _validate_and_convert_X(self, X):
        # accept numpy array or pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D array-like (n_samples, n_features)")
        return X

    def _build_model(self, input_dim, num_classes):
        """Create model instance and move to device."""
        model = MediumMLP(input_dim=input_dim, hidden=self.hidden, n_blocks=self.n_blocks,
                          num_classes=num_classes, dropout=self.dropout)
        model.to(self.device)
        return model

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Fit the model. Automatically infers input_dim from X if self.input_dim is None.
        """
        self._set_seed()

        # accept pandas DataFrame
        X = self._validate_and_convert_X(X)
        y = np.asarray(y)

        # infer input dim if needed
        if self.input_dim is None:
            self.input_dim = X.shape[1]

        # label encode
        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        num_classes = len(self.classes_)

        # scaler
        if self.scaler:
            self.scaler_ = StandardScaler()
            X_proc = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None
            X_proc = X.astype(np.float32)

        # train/val split if not provided
        if X_val is None or y_val is None:
            X_train, X_val_split, y_train, y_val_split = train_test_split(
                X_proc, y_enc, test_size=self.validation_fraction, random_state=self.seed,
                stratify=y_enc if num_classes > 1 else None)
        else:
            X_train = X_proc
            y_train = y_enc
            X_val_split = np.asarray(X_val)
            if self.scaler:
                X_val_split = self.scaler_.transform(X_val_split)
            y_val_split = np.asarray(self.label_encoder_.transform(y_val))

        # create datasets/loaders
        X_train = X_train.astype(np.float32)
        X_val_split = X_val_split.astype(np.float32)
        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long())
        val_ds = TensorDataset(torch.from_numpy(X_val_split), torch.from_numpy(y_val_split).long())

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=max(64, self.batch_size), shuffle=False)

        # build model lazily based on inferred input_dim
        self.model_ = self._build_model(self.input_dim, num_classes)

        optimizer = AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        scaler = torch.cuda.amp.GradScaler() if "cuda" in self.device else None

        best_val = float("inf")
        best_epoch = -1
        best_state = None
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, self.max_epochs + 1):
            epoch_start = time.time()
            # training
            self.model_.train()
            running_loss = 0.0
            n_seen = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        logits = self.model_(xb)
                        loss = criterion(logits, yb)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = self.model_(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
                bs = xb.shape[0]
                running_loss += float(loss.item()) * bs
                n_seen += bs

            train_loss = running_loss / max(1, n_seen)

            # validation
            self.model_.eval()
            val_loss_total = 0.0
            val_n = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    logits = self.model_(xb)
                    loss = criterion(logits, yb)
                    bs = xb.shape[0]
                    val_loss_total += float(loss.item()) * bs
                    val_n += bs
            val_loss = val_loss_total / max(1, val_n)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if self.verbose:
                epoch_time = time.time() - epoch_start
                print(f"Epoch {epoch:03d} | train_loss={train_loss:.5f} val_loss={val_loss:.5f} time={epoch_time:.2f}s")

            # early stopping
            if val_loss < best_val - 1e-8:
                best_val = val_loss
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
            elif (epoch - best_epoch) >= self.patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}, best_epoch={best_epoch}, best_val={best_val:.5f}")
                break

        # restore best weights if found
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.history_ = history
        return self

    def predict_proba(self, X):
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet. Call fit(X, y) first.")

        X = self._validate_and_convert_X(X)
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        X = X.astype(np.float32)

        ds = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(ds, batch_size=max(128, self.batch_size), shuffle=False)
        self.model_.eval()
        probs = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                logits = self.model_(xb)
                p = F.softmax(logits, dim=1).cpu().numpy()
                probs.append(p)
        probs = np.vstack(probs)
        return probs

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.label_encoder_.inverse_transform(idx)

    # sklearn compatibility
    def get_params(self, deep=True):
        # return stable params; input_dim may be None if not fitted yet
        return {k: getattr(self, k) for k in [
            "input_dim", "hidden", "n_blocks", "dropout", "lr", "weight_decay",
            "batch_size", "max_epochs", "patience", "validation_fraction",
            "device", "scaler", "seed", "verbose"
        ]}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    