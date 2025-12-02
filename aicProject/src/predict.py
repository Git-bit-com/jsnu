import os, json
import numpy as np
import pandas as pd
from joblib import load
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from src.utils import setup_logger, safe_read_csv

logger = setup_logger()
DATA_DIR = "data"
MODEL_DIR = "models"
OUT_DIR = "output"

def _ids_as_str(df):
    for c in ("user_id","book_id","DEPT","年级"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

def _to_int_safe(s):
    return s.astype(str).str.replace(r"\.0+$","", regex=True).astype(int)

def predict_top1(
    feat_path=os.path.join(DATA_DIR,"features.csv"),
    test_path=os.path.join(DATA_DIR,"test.csv"),
    train_path=os.path.join(DATA_DIR,"train.csv"),
    out_path=os.path.join(OUT_DIR,"submission.csv")
):
    logger.info("🚀 开始执行融合预测任务...")

    os.makedirs(OUT_DIR, exist_ok=True)
    feat = safe_read_csv(feat_path)
    test_users = safe_read_csv(test_path, usecols=["user_id"]).dropna()
    _ids_as_str(feat)
    _ids_as_str(test_users)

    feat = feat[feat["user_id"].isin(set(test_users["user_id"]))].copy()
    if feat.empty:
        raise RuntimeError("❌ features 中没有匹配到 test 用户的候选行。")

    with open(os.path.join(MODEL_DIR,"feature_names.json"), "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    X = feat[feature_cols].values

    logger.info("📦 加载 CatBoost / XGBoost / MLP 模型 ...")
    xgb = XGBClassifier()
    xgb.load_model(os.path.join(MODEL_DIR,"xgb_model.json"))
    cat = CatBoostClassifier()
    cat.load_model(os.path.join(MODEL_DIR,"cat_model.cbm"))
    mlp = load(os.path.join(MODEL_DIR,"mlp_model.pkl"))
    scaler = load(os.path.join(MODEL_DIR,"mlp_scaler.pkl"))

    logger.info("🔮 单模型预测 ...")
    px = xgb.predict_proba(X)[:,1]
    pc = cat.predict_proba(X)[:,1]
    pm = mlp.predict_proba(scaler.transform(X))[:,1]
    blend = {"xgb":0.45,"cat":0.45,"mlp":0.10}
    blend_path = os.path.join(MODEL_DIR,"blend.json")
    if os.path.exists(blend_path):
        try:
            with open(blend_path,"r",encoding="utf-8") as f:
                b2 = json.load(f)
                for k in blend:
                    if k in b2: blend[k]=float(b2[k])
        except Exception:
            pass

    logger.info("🎛️ 模型融合 ...")
    pred = blend["xgb"]*px + blend["cat"]*pc + blend["mlp"]*pm
    feat["pred_score"] = pred
    rec = (feat.sort_values(["user_id","pred_score"], ascending=[True, False])
              .groupby("user_id", as_index=False)
              .first()[["user_id","book_id"]])

    missing = sorted(set(test_users["user_id"]) - set(rec["user_id"]))
    if missing:
        logger.warning(f"⚠️ 候选缺失用户 {len(missing)}，启用兜底（院系/年级热门→全局热门）。")
        merged = safe_read_csv(train_path)
        _ids_as_str(merged)
        # 用户元数据
        meta = merged.sort_values("借阅时间").groupby("user_id")[["DEPT","年级"]].agg(lambda s: s.dropna().iloc[-1] if len(s.dropna()) else np.nan).reset_index()
        _ids_as_str(meta); meta = meta.set_index("user_id")
        # 用户历史
        hist = merged.groupby("user_id")["book_id"].apply(set).to_dict()
        # 组内热门
        grp = merged.groupby(["DEPT","年级","book_id"]).size().rename("cnt").reset_index()
        _ids_as_str(grp)
        # 全局热门
        gtop = merged["book_id"].value_counts().index.astype(str).tolist()

        extra = []
        for u in missing:
            dept = meta.loc[u, "DEPT"] if u in meta.index else np.nan
            grade = meta.loc[u, "年级"] if u in meta.index else np.nan
            seen = hist.get(u, set())
            g = grp[(grp["DEPT"]==str(dept)) & (grp["年级"]==str(grade))]
            picked = None
            if not g.empty:
                for b in g.sort_values("cnt", ascending=False)["book_id"]:
                    if b not in seen:
                        picked = b; break
            if picked is None:
                for b in gtop:
                    if b not in seen:
                        picked = b; break
            if picked is None:
                picked = gtop[0]
            extra.append({"user_id":u,"book_id":picked})
        rec = pd.concat([rec, pd.DataFrame(extra)], ignore_index=True)

    rec = rec.sort_values(["user_id"]).groupby("user_id", as_index=False).first()

    rec["user_id"] = _to_int_safe(rec["user_id"])
    rec["book_id"] = _to_int_safe(rec["book_id"])
    os.makedirs(OUT_DIR, exist_ok=True)
    rec.to_csv(out_path, index=False)
    logger.info(f"✅ 融合推荐完成，共 {len(rec)} 条记录（= 测试用户数）。已保存: {out_path}")

if __name__ == "__main__":
    predict_top1()
