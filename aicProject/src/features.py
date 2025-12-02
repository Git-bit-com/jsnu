import os
import argparse
import json
import numpy as np
import pandas as pd
from src.utils import setup_logger, safe_read_csv

logger = setup_logger()
DATA_DIR = "data"
MODEL_DIR = "models"


def _ids_as_str(df: pd.DataFrame):
    for c in ("user_id", "book_id", "DEPT", "年级"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def _normalize_columns(df: pd.DataFrame):
    df.columns = [c.strip().replace(" ", "") for c in df.columns]
    return df


def _latest_nonnull(s: pd.Series):
    s = s.dropna()
    return s.iloc[-1] if len(s) else np.nan


def generate_features(
    train_path=os.path.join(DATA_DIR, "train.csv"),
    cf_path=os.path.join(DATA_DIR, "cf_scores.csv"),
    out_path=os.path.join(DATA_DIR, "features.csv"),
    feature_names_path=os.path.join(MODEL_DIR, "feature_names.json"),
    test_path=os.path.join(DATA_DIR, "test.csv"),
):
    logger.info("📘 加载交互数据/候选数据...")
    inter = safe_read_csv(train_path)
    inter = _normalize_columns(inter)
    _ids_as_str(inter)
    inter["借阅时间"] = pd.to_datetime(inter["借阅时间"], errors="coerce")

    cf = safe_read_csv(cf_path)
    _ids_as_str(cf)

    label_map = {}
    if test_path is not None and os.path.exists(test_path):
        test = safe_read_csv(test_path)
        _ids_as_str(test)
        label_map = dict(
            zip(test["user_id"].astype(str), test["book_id"].astype(str))
        )
        logger.info(f"🔖 检测到离线真值 test.csv，含用户数={len(label_map)}")
    else:
        logger.info("ℹ️ 未检测到 test.csv，将使用完整历史构造特征（无泄露约束）。")

    user_meta = (
        inter.sort_values("借阅时间")
        .groupby("user_id")[["DEPT", "年级"]]
        .agg(_latest_nonnull)
        .reset_index()
    )
    _ids_as_str(user_meta)

    item_cnt = inter.groupby("book_id").size().rename("item_pop").astype(float)
    item_last = (
        inter.groupby("book_id")["借阅时间"].max().rename("item_last_time")
    )

    grp_cnt = (
        inter.groupby(["DEPT", "年级", "book_id"])
        .size()
        .rename("grp_pop")
        .astype(float)
        .reset_index()
    )
    _ids_as_str(grp_cnt)

    inter_sorted = inter.sort_values(["user_id", "借阅时间"])
    prefix_hist = {}
    last_time_by_user = {}

    empty_hist = {
        "items": set(),
        "authors": set(),
        "publishers": set(),
        "cat1": set(),
        "cat2": set(),
        "count": 0.0,
    }

    for u, dfu in inter_sorted.groupby("user_id"):
        dfu = dfu.copy()
        items = dfu["book_id"].astype(str).tolist()
        last_time_by_user[u] = dfu["借阅时间"].iloc[-1]

        if u in label_map and len(items) >= 1:
            prefix = dfu.iloc[:-1]
        else:
            prefix = dfu

        prefix_hist[u] = {
            "items": set(prefix["book_id"].astype(str).tolist()),
            "authors": set(
                str(x)
                for x in prefix.get("作者", pd.Series(dtype=str))
                .dropna()
                .tolist()
            ),
            "publishers": set(
                str(x)
                for x in prefix.get("出版社", pd.Series(dtype=str))
                .dropna()
                .tolist()
            ),
            "cat1": set(
                str(x)
                for x in prefix.get("一级分类", pd.Series(dtype=str))
                .dropna()
                .tolist()
            ),
            "cat2": set(
                str(x)
                for x in prefix.get("二级分类", pd.Series(dtype=str))
                .dropna()
                .tolist()
            ),
            "count": float(len(prefix)),
        }

    meta_cols = [
        c
        for c in ["作者", "出版社", "一级分类", "二级分类"]
        if c in inter.columns
    ]
    item_meta_dict = {}
    if meta_cols:
        item_meta = (
            inter_sorted.groupby("book_id")[meta_cols]
            .agg(_latest_nonnull)
            .reset_index()
        )
        _ids_as_str(item_meta)
        item_meta_dict = item_meta.set_index("book_id")[meta_cols].to_dict(
            "index"
        )
        logger.info(
            f"📚 已构建物品元数据字典，包含图书数={len(item_meta_dict)}，字段={meta_cols}"
        )
    else:
        logger.info("ℹ️ 未检测到 作者/出版社/分类 等字段，相关特征将为 0。")

    # ===== 候选集拼接基础特征 =====
    feat = cf.merge(user_meta, on="user_id", how="left")
    feat = feat.merge(item_cnt.to_frame(), on="book_id", how="left")
    feat = feat.merge(item_last.to_frame(), on="book_id", how="left")
    feat = feat.merge(
        grp_cnt, on=["DEPT", "年级", "book_id"], how="left"
    )

    feat["grp_pop"] = feat["grp_pop"].fillna(0.0)
    feat["item_pop"] = feat["item_pop"].fillna(0.0)

    ref_time = inter["借阅时间"].max()
    feat["item_recency_days"] = (
        ref_time - feat["item_last_time"]
    ).dt.days.clip(lower=0)
    feat["item_recency_days"] = (
        feat["item_recency_days"]
        .fillna(feat["item_recency_days"].max())
        .astype(float)
    )

    feat["log_item_pop"] = np.log1p(feat["item_pop"])
    feat["inv_item_recency"] = 1.0 / (1.0 + feat["item_recency_days"])

    ua, up, uc1, uc2, uhcnt, usince = [], [], [], [], [], []

    for row in feat.itertuples():
        u = row.user_id
        b = row.book_id

        hist = prefix_hist.get(u, empty_hist)

        meta_b = item_meta_dict.get(b)
        if meta_b is None:
            bauthor = bpub = bcat1 = bcat2 = ""
        else:
            bauthor = str(meta_b.get("作者", "") or "")
            bpub = str(meta_b.get("出版社", "") or "")
            bcat1 = str(meta_b.get("一级分类", "") or "")
            bcat2 = str(meta_b.get("二级分类", "") or "")

        ua.append(1.0 if bauthor and bauthor in hist["authors"] else 0.0)
        up.append(1.0 if bpub and bpub in hist["publishers"] else 0.0)
        uc1.append(1.0 if bcat1 and bcat1 in hist["cat1"] else 0.0)
        uc2.append(1.0 if bcat2 and bcat2 in hist["cat2"] else 0.0)
        uhcnt.append(hist.get("count", 0.0))

        last_t = last_time_by_user.get(u, ref_time)
        usince.append(float((ref_time - last_t).days))

    feat["same_author_hist"] = np.array(ua, dtype=float)
    feat["same_publisher_hist"] = np.array(up, dtype=float)
    feat["same_cat1_hist"] = np.array(uc1, dtype=float)
    feat["same_cat2_hist"] = np.array(uc2, dtype=float)
    feat["user_hist_count"] = np.array(uhcnt, dtype=float)
    feat["user_days_since_last"] = np.array(usince, dtype=float)

    feature_cols = [
        "cf_score",
        "log_item_pop",
        "inv_item_recency",
        "grp_pop",
        "same_author_hist",
        "same_publisher_hist",
        "same_cat1_hist",
        "same_cat2_hist",
        "user_hist_count",
        "user_days_since_last",
    ]

    feat_out = feat[["user_id", "book_id"] + feature_cols].copy()
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    feat_out.to_csv(out_path, index=False)

    with open(feature_names_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    logger.info(
        f"✅ 特征生成完成: {out_path} (rows={len(feat_out)}, cols={len(feature_cols)+2})"
    )
    logger.info(
        f"📊 覆盖用户: {feat_out['user_id'].nunique()}, 书籍: {feat_out['book_id'].nunique()}"
    )
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    generate_features()
