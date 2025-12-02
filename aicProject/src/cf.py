import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from src.utils import setup_logger, safe_read_csv

logger = setup_logger()
DATA_DIR = "data"


def _ids_as_str(df: pd.DataFrame):
    for c in ("user_id", "book_id", "DEPT", "年级"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def _normalize_columns(df: pd.DataFrame):
    df.columns = [c.strip().replace(" ", "") for c in df.columns]
    return df


def _ensure_target_users(
    train: pd.DataFrame, path=os.path.join(DATA_DIR, "test_users_all.csv")
):

    if os.path.exists(path):
        tu = safe_read_csv(path)
        _ids_as_str(tu)
        users = sorted(set(tu["user_id"].astype(str)))
        logger.info(f"✅ 从 {path} 载入目标用户数={len(users)}")
        return users

    # 所有在交互表中出现过的用户
    last_time = train.groupby("user_id")["借阅时间"].max().reset_index()
    last_time = last_time.sort_values("借阅时间", ascending=False)
    users = last_time["user_id"].astype(str).tolist()
    pd.DataFrame({"user_id": users}).to_csv(path, index=False)
    logger.info(
        f"✅ 未找到目标用户文件，已自动根据交互数据生成: {path}（用户数={len(users)}）"
    )
    return users


def _build_user_splits(train: pd.DataFrame, target_users):
    g = train.sort_values(["user_id", "借阅时间"]).groupby("user_id")
    user_full_items = {}
    user_prefix_items = {}
    label_map = {}
    target_users = set(target_users)

    for u, dfu in g:
        items = dfu["book_id"].tolist()
        user_full_items[u] = set(items)
        if u in target_users and len(items) >= 1:
            label_map[u] = str(items[-1])
            user_prefix_items[u] = set(items[:-1])
        else:
            user_prefix_items[u] = set(items)
    return user_prefix_items, user_full_items, label_map


def build_cf_scores(
    topk=300,
    train_path=os.path.join(DATA_DIR, "train.csv"),
    test_users_path=os.path.join(DATA_DIR, "test_users_all.csv"),
    out_path=os.path.join(DATA_DIR, "cf_scores.csv"),
):

    train = safe_read_csv(train_path)
    train = _normalize_columns(train)
    _ids_as_str(train)

    if "借阅时间" in train.columns:
        train["借阅时间"] = pd.to_datetime(train["借阅时间"], errors="coerce")

    target_users = _ensure_target_users(train, path=test_users_path)

    user_prefix_items, user_full_items, label_map = _build_user_splits(
        train, target_users
    )

    item_users = defaultdict(set)
    for u, items in user_full_items.items():
        for b in items:
            item_users[b].add(u)

    grp_cnt = (
        train.groupby(["DEPT", "年级", "book_id"])
        .size()
        .rename("cnt")
        .reset_index()
    )
    _ids_as_str(grp_cnt)

    global_hot = train["book_id"].value_counts().index.astype(str).tolist()

    user_meta = (
        train.sort_values("借阅时间")
        .groupby("user_id")[["DEPT", "年级"]]
        .agg(lambda s: s.dropna().iloc[-1] if len(s.dropna()) else np.nan)
    )

    rows = []
    for u in target_users:
        uhist = user_prefix_items.get(u, set())
        cand_scores = Counter()

        for b in uhist:
            for v in item_users.get(b, ()):
                if v == u:
                    continue
                vset = user_full_items.get(v, set())
                if not vset:
                    continue
                inter = len(uhist & vset)
                if inter == 0:
                    continue
                sim = inter / (
                    (len(uhist) or 1) ** 0.5 * (len(vset) or 1) ** 0.5
                )
                for nb in vset:
                    if nb in uhist:
                        continue
                    cand_scores[nb] += sim

        dept = str(user_meta.loc[u, "DEPT"]) if u in user_meta.index else np.nan
        grade = str(user_meta.loc[u, "年级"]) if u in user_meta.index else np.nan
        gdf = grp_cnt[(grp_cnt["DEPT"] == dept) & (grp_cnt["年级"] == grade)]
        if not gdf.empty:
            for _, r in gdf.iterrows():
                if r["book_id"] not in uhist:
                    cand_scores[r["book_id"]] += 0.05 * float(r["cnt"])

        if not cand_scores:
            for b in global_hot:
                if b not in uhist:
                    cand_scores[b] += 1.0

        for b, s in cand_scores.most_common(topk):
            rows.append((u, str(b), float(s)))

    cf = pd.DataFrame(rows, columns=["user_id", "book_id", "cf_score"])
    cf.to_csv(out_path, index=False)
    logger.info(
        f"✅ CF 候选生成：{len(cf)} 行，覆盖用户={cf['user_id'].nunique()} / 目标用户={len(target_users)}，TopK/用户≈{topk} → {out_path}"
    )
    return out_path

def generate_cf_scores(topk=300, **kwargs):
    return build_cf_scores(topk=topk, **kwargs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=300)
    args = ap.parse_args()
    build_cf_scores(topk=args.topk)
