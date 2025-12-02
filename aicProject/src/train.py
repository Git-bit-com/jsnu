import os, json, argparse, warnings
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from src.utils import setup_logger, safe_read_csv
from src.cf import build_cf_scores
from src.features import generate_features

warnings.filterwarnings("ignore")
logger = setup_logger()
DATA_DIR = "data"
MODEL_DIR = "models"
OUT_DIR = "output"


def _ids_as_str(df: pd.DataFrame):
    for c in ("user_id", "book_id"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def _normalize_columns(df: pd.DataFrame):
    df.columns = [c.strip().replace(" ", "") for c in df.columns]
    return df


def _to_int_safe(s: pd.Series):
    return s.astype(str).str.replace(r"\.0+$", "", regex=True).astype(int)


def _prepare_validation(
    train_path=os.path.join(DATA_DIR, "train.csv"),
    test_users_path=os.path.join(DATA_DIR, "test_users_all.csv"),
    test_path=os.path.join(DATA_DIR, "test.csv"),
):
    inter = safe_read_csv(train_path)
    inter = _normalize_columns(inter)
    _ids_as_str(inter)
    inter["借阅时间"] = pd.to_datetime(inter["借阅时间"], errors="coerce")

    last_time = inter.groupby("user_id")["借阅时间"].max().reset_index()
    last_time = last_time.sort_values("借阅时间", ascending=False)
    test_users = last_time["user_id"].astype(str).tolist()

    pd.DataFrame({"user_id": test_users}).to_csv(test_users_path, index=False)
    logger.info(f"✅ 生成目标用户列表: {test_users_path}（用户数={len(test_users)}）")

    g = inter.sort_values(["user_id", "借阅时间"]).groupby("user_id")
    rows = [{"user_id": u, "book_id": str(dfu["book_id"].iloc[-1])} for u, dfu in g]
    test = pd.DataFrame(rows)
    test.to_csv(test_path, index=False)
    logger.info(f"✅ 生成离线真值: {test_path}（{len(test)} 条，对应用户数={test['user_id'].nunique()}）")

    return test_users


def _make_train_table(features_path, test_path):
    feat = safe_read_csv(features_path)
    _ids_as_str(feat)
    test = safe_read_csv(test_path)
    _ids_as_str(test)

    pos = set(zip(test["user_id"], test["book_id"]))
    feat["label"] = feat.apply(
        lambda r: 1 if (r["user_id"], r["book_id"]) in pos else 0, axis=1
    )

    def sample_user(dfu: pd.DataFrame):
        posu = dfu[dfu["label"] == 1]
        negu = dfu[dfu["label"] == 0]
        k = min(len(negu), 60)
        if k > 0:
            negu = negu.sample(n=k, random_state=42)
        return pd.concat([posu, negu], ignore_index=True)

    feat = (
        feat.groupby("user_id", group_keys=False)
        .apply(sample_user)
        .reset_index(drop=True)
    )
    logger.info(
        f"🧰 训练样本：{feat['label'].sum()} 正 / {len(feat) - feat['label'].sum()} 负 / 共 {len(feat)}"
    )
    return feat


def train_model():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    test_users = _prepare_validation()
    build_cf_scores(topk=500)
    features_path = generate_features()
    feat = _make_train_table(features_path, os.path.join(DATA_DIR, "test.csv"))
    with open(
        os.path.join(MODEL_DIR, "feature_names.json"), "r", encoding="utf-8"
    ) as f:
        feature_cols = json.load(f)

    X = feat[feature_cols].values.astype(float)
    y = feat["label"].values.astype(int)

    users = feat["user_id"].unique()
    tr_users, va_users = sklearn.model_selection.train_test_split(
        users, test_size=0.2, random_state=42
    )
    tr = feat[feat["user_id"].isin(set(tr_users))].copy()
    va = feat[feat["user_id"].isin(set(va_users))].copy()
    Xtr, ytr = tr[feature_cols].values, tr["label"].values
    Xva, yva = va[feature_cols].values, va["label"].values

    class_weight = None
    if len(np.unique(ytr)) == 2:
        classes = np.array([0, 1])
        w = compute_class_weight(class_weight="balanced", classes=classes, y=ytr)
        class_weight = {0: float(w[0]), 1: float(w[1])}

    logger.info("🚀 训练 XGBoost ...")
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        tree_method="hist",
        random_state=42,
    )
    xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

    logger.info("🚀 训练 CatBoost ...")
    cat = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        random_state=42,
        verbose=False,
        class_weights=None
        if class_weight is None
        else [class_weight[0], class_weight[1]],
    )
    cat.fit(Pool(Xtr, ytr), eval_set=Pool(Xva, yva), verbose=False)

    logger.info("🚀 训练 MLP ...")
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=50,
        random_state=42,
    )
    mlp.fit(Xtr_s, ytr)

    os.makedirs(MODEL_DIR, exist_ok=True)
    xgb.save_model(os.path.join(MODEL_DIR, "xgb_model.json"))
    cat.save_model(os.path.join(MODEL_DIR, "cat_model.cbm"))
    dump(mlp, os.path.join(MODEL_DIR, "mlp_model.pkl"))
    dump(scaler, os.path.join(MODEL_DIR, "mlp_scaler.pkl"))

    logger.info("✅ 模型已保存到 models/ 目录")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    train_model()
