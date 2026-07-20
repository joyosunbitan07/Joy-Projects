Project 2 submission

## Repository Deliverables
* `data_processing.py`: Custom streaming parser that structures raw data, cleans missing entries, and splits datasets.
* `train_models.py`: Model training pipeline that fits the LightGBM classifier and handles data skew.
* `test_predict.py`: Deployment verification script to run sample inferences.
* `malware_lgb_model.pkl`: Trained LightGBM model binary.
* `feature_importance.png`: Top 15 features driving model decisions.
* `Project 2 Documentation.pdf`: Full technical summary report and operational instructions.

## Replication Instructions
To run the source code end-to-end, please ensure the following source files are placed into this folder directory alongside the scripts:
1. `feature_name_to_number_mapping.csv`
2. `project2_raw_data/` (The folder containing the 45 raw continuous text log files)

Execution Order:
1. Process data: `python data_processing.py`
2. Train & evaluate model: `python train_models.py`
3. Run sample inference: `python test_predict.py`

## Replication & Data Architecture
Because raw telemetry logs are massive and belong to an external pipeline, the raw text logs (`project2_raw_data/`) are omitted from this repository to maintain clean version control.

* **Production Inference Test:** The deployment script (`test_predict.py`) is fully self-contained and utilizes an internal mock telemetry payload. You can run `python test_predict.py` immediately to see the model execute predictions without downloading the raw dataset.
* **Full Retraining Pipeline:** To run `data_processing.py` and `train_models.py` from scratch, the raw continuous log files must be placed back into a `./project2_raw_data/` directory.
