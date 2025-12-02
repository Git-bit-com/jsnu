from src.utils import safe_read_csv, ensure_dir, setup_logger
import pandas as pd
import re

logger = setup_logger()

def _normalize_whitespace(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.replace('\u3000', ' ')     # 全角空格
    s = s.replace('\u200b', '')      # 零宽空格
    s = s.replace('\ufeff', '')      # BOM
    s = s.strip()
    return re.sub(r'\s+', ' ', s)

def _strip_all_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].astype("string").apply(lambda x: _normalize_whitespace(x) if pd.notna(x) else x)
        df.loc[df[c] == "", c] = pd.NA
    return df

def _robust_to_datetime(series: pd.Series):
    s = series.astype("string").apply(lambda x: _normalize_whitespace(x) if pd.notna(x) else x)
    missing_tokens = {"", "nan", "na", "null", "none", "None", "未还", "未 还", "未借", "未 借"}
    s = s.apply(lambda x: pd.NA if pd.isna(x) or str(x).strip() in missing_tokens else x)
    s = s.apply(lambda x: re.sub(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})(\d{1,2}:\d{2}:\d{2})", r"\1 \2", x)
                if isinstance(x, str) else x)
    return pd.to_datetime(s, errors="coerce")

def load_and_merge_data():
    user = safe_read_csv("data/user.csv")
    item = safe_read_csv("data/item.csv")
    inter = safe_read_csv("data/inter_preliminary.csv")

    def clean_cols(df):
        df = df.copy()
        df.columns = [_normalize_whitespace(c) if isinstance(c, str) else c for c in df.columns]
        return df

    user, item, inter = map(clean_cols, [user, item, inter])
    logger.info(f"✅ 原始数据载入：user {user.shape}, item {item.shape}, inter {inter.shape}")

    user = _strip_all_string_columns(user)
    item = _strip_all_string_columns(item)
    inter = _strip_all_string_columns(inter)

    if "user_id" not in inter.columns and "借阅人" in inter.columns:
        inter = inter.rename(columns={"借阅人": "user_id"})
    if "book_id" not in inter.columns and "图书ID" in inter.columns:
        inter = inter.rename(columns={"图书ID": "book_id"})
    if "借阅人" not in user.columns and "user_id" in user.columns:
        user = user.rename(columns={"user_id": "借阅人"})
    if "book_id" not in item.columns:
        for cand in ["图书ID", "书号", "bookID"]:
            if cand in item.columns:
                item = item.rename(columns={cand: "book_id"})
                break

    for df_, col in [(inter, "user_id"), (inter, "book_id"), (user, "借阅人"), (item, "book_id")]:
        if col in df_.columns:
            df_[col] = df_[col].astype("string").apply(lambda x: _normalize_whitespace(x) if pd.notna(x) else x)

    for col in ["借阅时间", "还书时间", "续借时间", "borrow_date", "return_date"]:
        if col in inter.columns:
            inter[col] = _robust_to_datetime(inter[col])

    df = inter.merge(user, left_on="user_id", right_on="借阅人", how="left")
    df = df.merge(item, on="book_id", how="left")

    ensure_dir("data")
    df.to_csv("data/train.csv", index=False, encoding="utf-8")
    logger.info(f"✅ 数据清洗与合并完成 → data/train.csv ({len(df)} 行)")

    logger.info("样例记录（前 5 行）：")
    logger.info(df.head(5).to_dict(orient="records"))

    return df

if __name__ == "__main__":
    df = load_and_merge_data()
    logger.info("🎯 数据预处理完成！")
