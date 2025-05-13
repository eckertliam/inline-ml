from typing import Tuple
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

TRAINING_FEATURES = [
    "callee_instruction_count",
    "callee_total_calls",
    "callee_load_store_ratio",
    "caller_instruction_count",
    "callee_arg_count",
]


def prep_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess the data to train the model on
    Convert bools to ints, drop columns we filtered during feature selection.

    Args:
        data (pd.DataFrame): The data to preprocess.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: The preprocessed x and y data.
    """
    # keep only the target (llvm_inlining_decision) and training features
    filtered_df = data[["llvm_inlining_decision"] + TRAINING_FEATURES]
    # drop NaNs
    nan_df = filtered_df.dropna()
    # get the features
    x = nan_df[TRAINING_FEATURES]
    # get the target and convert bool to int
    y = nan_df["llvm_inlining_decision"].astype(int)

    return (x, y)


if __name__ == "__main__":

    data = pd.read_csv("data/csv/data.csv")

    x, y = prep_data(data)

    # get the training and test data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    param_dist = {
        # number of trees to fit
        # NOTE: more trees can lead to overfitting
        "n_estimators": [100, 200, 300],
        # max depth for each tree
        # NOTE: again, more depth can lead to overfitting
        "max_depth": [3, 4, 5],
        # learning rate
        # NOTE: each tree corrects the previous by this fraction, lower leads to more trees and more robustness
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        # subsample ratio of training instances for each tree
        "subsample": [0.6, 0.8, 1.0],
        # subsample ratio of columns when constructing each tree
        "colsample_bytree": [0.6, 0.8, 1.0],
    }

    # initialize the base model
    base_model = XGBClassifier(eval_metric="logloss")

    # initialize the randomized search
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=50,  # the number of param combos to try
        scoring="f1",  # optimization metric
        cv=5,  # number of folds for cross val
        verbose=1,  # 0 no prog 1 prog
        random_state=42,  # NOTE: always 42 for reproducability
        n_jobs=-1,  # uses all cores
    )

    # fit the model
    search.fit(x_train, y_train)

    # get the best model
    best_model = search.best_estimator_
    # please the type checker
    assert isinstance(best_model, XGBClassifier)

    # eval the model
    y_pred = best_model.predict(x_test)
    print(classification_report(y_test, y_pred))

    # save the model
    best_model.save_model("inline_model.json")
