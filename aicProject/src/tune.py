import os
import json
import argparse
import numpy as np
import pandas as pd
from joblib import load
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from src.utils import setup_logger, safe_read_csv

logger = setup_logger()
DATA_DIR = "data"
MODEL_DIR = "models"


def _ids_as_str(df: pd.DataFrame):
    for c in ("user_id", "book_id"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def tune_blend(step: float = 0.1):
    features_path = os.path.join(DATA_DIR, "features.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    if not os.path.exists(features_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"❌ 需要先训练好模型并生成 {features_path} 与 {test_path} 后才能进行权重搜索。"
        )

    feat = safe_read_csv(features_path)
    _ids_as_str(feat)
    test = safe_read_csv(test_path)
    _ids_as_str(test)

    label_map = dict(zip(test["user_id"], test["book_id"]))
    feat = feat[feat["user_id"].isin(label_map.keys())].copy()

    with open(
        os.path.join(MODEL_DIR, "feature_names.json"), "r", encoding="utf-8"
    ) as f:
        feature_cols = json.load(f)

    X = feat[feature_cols].values.astype(float)

    logger.info("📦 加载模型以进行权重搜索 ...")
    xgb = XGBClassifier()
    xgb.load_model(os.path.join(MODEL_DIR, "xgb_model.json"))
    cat = CatBoostClassifier()
    cat.load_model(os.path.join(MODEL_DIR, "cat_model.cbm"))
    mlp = load(os.path.join(MODEL_DIR, "mlp_model.pkl"))
    scaler = load(os.path.join(MODEL_DIR, "mlp_scaler.pkl"))

    logger.info("🔮 预先计算各模型的预测概率 ...")
    px = xgb.predict_proba(X)[:, 1]
    pc = cat.predict_proba(X)[:, 1]
    pm = mlp.predict_proba(scaler.transform(X))[:, 1]

    feat = feat[["user_id", "book_id"]].copy()
    feat["px"] = px
    feat["pc"] = pc
    feat["pm"] = pm

    users = feat["user_id"].unique()
    logger.info(
        f"🔍 参与权重搜索的用户数: {len(users)}, 候选样本数: {len(feat)}"
    )

    steps = np.arange(0.0, 1.0 + 1e-8, step)
    best_f1 = -1.0
    best_w = (0.45, 0.45, 0.10)
    best_detail = None

    for wx in steps:
        for wc in steps:
            wm = 1.0 - wx - wc
            if wm < 0 or wm > 1:
                continue

            feat["score"] = (
                wx * feat["px"] + wc * feat["pc"] + wm * feat["pm"]
            )

            idx = feat.groupby("user_id")["score"].idxmax()
            top = feat.loc[idx, ["user_id", "book_id"]].copy()
            top["true_book"] = top["user_id"].map(label_map)

            hits = (top["book_id"] == top["true_book"]).sum()
            p = hits / len(top) if len(top) > 0 else 0.0
            r = hits / len(label_map) if len(label_map) > 0 else 0.0
            f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)

            if f1 > best_f1:
                best_f1 = f1
                best_w = (float(wx), float(wc), float(wm))
                best_detail = (p, r, f1)
                logger.info(
                    f"✨ 当前最好权重: xgb={wx:.2f}, cat={wc:.2f}, mlp={wm:.2f} -> F1={f1:.4f} (P={p:.4f}, R={r:.4f})"
                )

    blend = {"xgb": best_w[0], "cat": best_w[1], "mlp": best_w[2]}
    os.makedirs(MODEL_DIR, exist_ok=True)
    blend_path = os.path.join(MODEL_DIR, "blend.json")
    with open(blend_path, "w", encoding="utf-8") as f:
        json.dump(blend, f, ensure_ascii=False, indent=2)

    logger.info("✅ 最优融合权重已写入 models/blend.json ：")
    logger.info(blend)
    if best_detail:
        p, r, f1 = best_detail
        logger.info(
            f"    对应离线指标: P={p:.4f}, R={r:.4f}, F1={f1:.4f}"
        )

    print("\n=== 最优融合权重 ===")
    print(json.dumps(blend, ensure_ascii=False, indent=2))
    print(f"\n对应 F1 ≈ {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        type=float,
        default=0.1,
        help="搜索步长，例如 0.1 -> 权重取 0.0,0.1,...,1.0 （总和=1）",
    )
    args = parser.parse_args()
    tune_blend(step=args.step)
