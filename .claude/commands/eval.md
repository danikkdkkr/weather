Evaluate a trained weather forecasting model and report per-variable metrics.

Ask the user which model file and checkpoint to evaluate (default: `best_model.pt` with the architecture from `weather_LSTM.py`).

Then:
1. Load the checkpoint and reconstruct the model
2. Run inference on the held-out test set using `evaluate_full_test()` from `training.py`
3. Report **MAE and RMSE per target variable** for each forecast horizon day (+1, +2, +3)
4. Highlight which variables and horizon days have the highest error
5. Compare against a naive baseline (persist last observed value) to show model skill

Format the output as a markdown table.
