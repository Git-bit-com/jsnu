import pandas as pd
import os
from src.utils import setup_logger

logger = setup_logger()
DATA_DIR = "data"
OUT_DIR = "output"

def evaluate_f1(pred_path=os.path.join(OUT_DIR,"submission.csv"),
                test_path=os.path.join(DATA_DIR,"test.csv")):
    logger.info(" 运行 evaluate.py")
    logger.info(" 读取预测与真实文件...")
    pred = pd.read_csv(pred_path)
    truth = pd.read_csv(test_path, usecols=["user_id","book_id"])
    pred["user_id"] = pred["user_id"].astype(int)
    pred["book_id"] = pred["book_id"].astype(int)
    truth["user_id"] = truth["user_id"].astype(int)
    truth["book_id"] = truth["book_id"].astype(int)
    truth_users = set(truth["user_id"].unique())
    pred = pred[pred["user_id"].isin(truth_users)].copy()
    truth_set = truth.groupby("user_id")["book_id"].apply(set).to_dict()
    hits = sum(int(row.book_id in truth_set.get(row.user_id, set())) for row in pred.itertuples())
    p = hits / max(len(pred), 1)
    r = hits / max(len(truth_users), 1)
    f1 = 0.0 if (p+r)==0 else 2*p*r/(p+r)

    logger.info("开始计算 Precision / Recall / F1 ...")
    logger.info(f"✅ Precision = {p:.4f}")
    logger.info(f"✅ Recall    = {r:.4f}")
    logger.info(f"✅ F1 Score  = {f1:.4f}")

    print("\n=== 模型评估结果 ===")
    print(f"Precision: {p:.4f}\nRecall:    {r:.4f}\nF1:        {f1:.4f}")
    return f1

if __name__ == "__main__":
    evaluate_f1()
