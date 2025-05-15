import shutil
import tempfile
from typing import List, Union
from pathlib import Path

from xgboost import XGBClassifier
from pipeline import rec_compile
from ir2df import mod2df
from train_model import TRAINING_FEATURES
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def check_llvm_tools():
    for tool in ["opt", "clang"]:
        if not shutil.which(tool):
            raise RuntimeError(
                f"Required LLVM tool {tool} not found in PATH.\n"
                f"This tool is required to run this program.\n"
                f"Please follow the user guide to install the required tools."
            )


def extract_callsites(dir_path: Path) -> Union[pd.DataFrame, str]:
    if not dir_path.exists():
        return "Directory does not exist"

    # compile the C files in the directory
    # returns a list of LLVM IR strings
    ir_strings: List[str] = rec_compile([dir_path])

    if len(ir_strings) == 0:
        return "No files were compiled"

    # extract the feature vectors from the LLVM IR strings and concat the results into a single dataframe
    df = pd.concat([mod2df(ir_string) for ir_string in ir_strings], ignore_index=True)

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
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Inlined", "Not Inlined"]
    )

    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title("Model vs LLVM Inlining Decisions")

    # make a temporary file to store the image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        tmp_path = Path(temp_file.name)

    fig.savefig(tmp_path, bbox_inches="tight")

    # close the figure
    plt.close(fig)

    # return the path to the image
    return str(tmp_path)
