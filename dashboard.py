"""
dashboard.py
-----------
This file is responsible for the dashboard that allows the user to open a C file
from disk and predict whether each function will be inlined.

Usage:
    poetry run python dashboard.py
"""

from pathlib import Path
from typing import Optional
from xgboost import XGBClassifier
from dashboard_utils import extract_callsites, make_confusion_matrix, predict_inlining
import pandas as pd
import dearpygui.dearpygui as dpg

selected_dir_path: Optional[str] = None

model = XGBClassifier()
model.load_model("inline_model.json")

def handle_directory_selection(sender, app_data):
    selected_dir = app_data['file_path_name']
    result = extract_callsites(Path(selected_dir))

    with dpg.window(label="Results", width=900, height=500, pos=(10, 10)):
        if isinstance(result, str):
            dpg.add_text(f"Error: {result}")
        elif isinstance(result, pd.DataFrame):
            df = result
            df_preds = predict_inlining(model, df)

            headers = list(df_preds.columns)
            with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp):
                for h in headers:
                    dpg.add_table_column(label=h)
                for i, row in df_preds.iterrows():
                    with dpg.table_row():
                        for val in row:
                            dpg.add_text(str(val))
                            
    # make the confusion matrix
    confusion_matrix_path = make_confusion_matrix(df_preds)
    width, height, channels, data = dpg.load_image(confusion_matrix_path)

    with dpg.texture_registry():
        texture_id = dpg.add_static_texture(width, height, data)

    # Display the image inside the same window
    with dpg.window(label="Confusion Matrix", width=width + 20, height=height + 60, pos=(10, 10)):
        dpg.add_image(texture_id)
    

dpg.create_context()
dpg.create_viewport(title='Inline-ML Dashboard', width=1000, height=600)

with dpg.window(label="Inline-ML", width=500, height=150):
    dpg.add_text("Select a C project directory to analyze:")
    dpg.add_button(label="Select Directory", callback=lambda: dpg.show_item("file_dialog_id"))

with dpg.file_dialog(directory_selector=True, show=False, callback=handle_directory_selection,
                     id="file_dialog_id", width=700, height=400):
    dpg.add_file_extension(".*")
    

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()