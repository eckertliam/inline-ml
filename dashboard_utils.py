import tempfile
from typing import Union
from pathlib import Path

from xgboost import XGBClassifier
from pipeline import rec_compile
from ir2df import mod2df
from train_model import TRAINING_FEATURES
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def extract_callsites(dir_path: Path) -> Union[pd.DataFrame, str]:
    if not dir_path.exists():
        return "Directory does not exist"

    # compile the files in the directory
    # rec_compile is used in pipeline.py to compile a list of directories so we just wrap the dir_path in a list
    ll_files = rec_compile([dir_path])

    if len(ll_files) == 0:
        return "No files were compiled"

    # concat the ll files into a single dataframe
    df = pd.concat([mod2df(ll_file) for ll_file in ll_files], ignore_index=True)
    # delete the ll files
    for ll_file in ll_files:
        ll_file.unlink()

    if df.empty:
        return "No callsites were extracted"

    # return the dataframe
    return df


def predict_inlining(model: XGBClassifier, df: pd.DataFrame) -> pd.DataFrame:
    # get the features
    df_x = df[TRAINING_FEATURES]
    # predict
    preds = model.predict(df_x).astype(bool)

    df_preds = df[["callee_name", "caller_name", "llvm_inlining_decision"]].copy()
    df_preds["model_inlining_decision"] = preds

    return df_preds


def make_confusion_matrix(df: pd.DataFrame) -> str:
    matplotlib.use("Agg")
    
    y_true = df["llvm_inlining_decision"].astype(bool)
    y_pred = df["model_inlining_decision"].astype(bool)

    cm = confusion_matrix(y_true, y_pred, labels=[True, False])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Inlined", "Not Inlined"])

    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title("Model vs LLVM Inlining Decisions")

    # Save to temp image file
    tmp_path = Path(tempfile.gettempdir()) / "confusion_matrix.png"
    fig.savefig(tmp_path, bbox_inches="tight")
    plt.close(fig)

    return str(tmp_path)