from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import numpy as np
import torch
import re
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Data Loading and Cleaning

df = pd.read_csv("EIA930_BALANCE_2026_Jul_Dec.csv")

# EIA CSVs often store numeric columns with thousands separators (e.g. "12,345"),
# which pandas reads as strings unless told otherwise. Force numeric conversion:
for col in ["Demand (MW)", "Demand Forecast (MW)"]:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )

df["UTC Time at End of Hour"] = pd.to_datetime(df["UTC Time at End of Hour"])

# Picking a single balancing authority

print(df["Balancing Authority"].value_counts())

BA = "MISO"   # change to whichever BA you want, e.g. "MISO", "ERCO", "CISO"
df_ba = df[df["Balancing Authority"] == BA].copy()

df_ba = df_ba.sort_values("UTC Time at End of Hour")

# Check for duplicate timestamps before assuming one row per hour

dupe_count = df_ba["UTC Time at End of Hour"].duplicated().sum()
print("Duplicate timestamps:", dupe_count)

if dupe_count > 0:
    # if duplicates exist, keep the most recently reported row per hour
    # (EIA sometimes has both a preliminary and a later corrected submission)
    df_ba = df_ba.drop_duplicates(subset="UTC Time at End of Hour", keep="last")

df_ba = df_ba.sort_values("UTC Time at End of Hour").reset_index(drop=True)

# Drop rows with missing demand (can happen with reporting gaps)

before = len(df_ba)
df_ba = df_ba.dropna(subset=["Demand (MW)"]).reset_index(drop=True)
print(f"Dropped {before - len(df_ba)} rows with missing Demand (MW)")

# Build series + timestamps

series = df_ba["Demand (MW)"].astype(float).tolist()
timestamps = df_ba["UTC Time at End of Hour"].tolist()

print("Total points:", len(series))

# Sanity check: confirm no missing hours

full_range = pd.date_range(
    df_ba["UTC Time at End of Hour"].min(),
    df_ba["UTC Time at End of Hour"].max(),
    freq="H"
)
missing = full_range.difference(df_ba["UTC Time at End of Hour"])
print("Missing hours:", len(missing), "/", len(full_range))

# Pipeline

WINDOW = 50  # 2 days of hourly context

# first WINDOW points serve as a seed
# they are only ever used as initial context never predicted or scored
history = series[:WINDOW].copy()

# every remaining point in the dataset gets predicted and scored
eval_series = series[WINDOW:]
eval_timestamps = timestamps[WINDOW:]

print("Seed window size:", len(history))
print("Points to predict/score:", len(eval_series))

predictions = []
raw_logs = []
fallback_count = 0

# Prediction Function
def predict_next(history, target_timestamp):
    global fallback_count

    hist_arr = np.array(history, dtype=float)
    last_val = hist_arr[-1]

    # predict deltas instead of raw levels (an easier signal to extrapolate)
    deltas = np.diff(hist_arr)
    mu, sigma = deltas.mean(), deltas.std() + 1e-6
    norm_deltas = (deltas - mu) / sigma

    history_text = "\n".join(f"{v:.3f}" for v in norm_deltas[-WINDOW:])
    target_desc = f"{target_timestamp.strftime('%A')}, hour {target_timestamp.hour:02d}:00 UTC"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a time series forecasting assistant for hourly "
                "electricity load. You will be given a chronological sequence "
                "of normalized HOUR-OVER-HOUR CHANGES (not levels), zero mean "
                "unit variance, most recent last. Predict the next change. "
                "Respond with ONLY the number, nothing else."
            )
        },
        {
            "role": "user",
            "content": f"{history_text}\n\nNext value corresponds to: {target_desc}\nNext change:"
        }
    ]

    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=12, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

    fell_back = False
    match = re.search(r"-?\d+\.?\d*", text)
    if match:
        try:
            norm_delta_pred = float(match.group())
            delta_pred = norm_delta_pred * sigma + mu
            # sanity bound the delta itself, not the level
            if abs(delta_pred) > 6 * sigma + abs(mu):
                delta_pred = 0.0
                fell_back = True
            value = last_val + delta_pred
        except ValueError:
            value = last_val
            fell_back = True
    else:
        value = last_val
        fell_back = True

    if fell_back:
        fallback_count += 1
    raw_logs.append({"target_timestamp": target_timestamp, "decoded_text": text,
                      "parsed_value": value, "fell_back": fell_back})
    return value

# Walk Forward Forecasting

for i, actual in enumerate(eval_series):
    pred = predict_next(history, eval_timestamps[i])
    predictions.append(pred)

    # slide forward using the real observed value and not the model's own prediction
    history.append(actual)
    history = history[-WINDOW:]

    if i % 500 == 0:
        print(f"Step {i}/{len(eval_series)} | fallback rate so far: {fallback_count/(i+1):.1%}")

print("="*40)
print(f"Fallback rate: {fallback_count/len(eval_series):.1%}")
print("="*40)
print("Sample of 10 raw model outputs:")
for row in raw_logs[:10]:
    print(row)

# LLM Performance

mae = mean_absolute_error(eval_series, predictions)
rmse = np.sqrt(mean_squared_error(eval_series, predictions))
mape = np.mean(np.abs((np.array(eval_series) - np.array(predictions)) / np.array(eval_series))) * 100
r2 = r2_score(eval_series, predictions)

print("="*40)
print("LLM FORECAST")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"MAPE : {mape:.2f}%")
print(f"R²   : {r2:.4f}")
print("="*40)

# Naive baseline (predict previous actual value)
# First naive prediction uses the last point of the seed window
# every subsequent one uses the previous actual in eval_series

naive_preds = [series[WINDOW - 1]] + eval_series[:-1]

naive_mae = mean_absolute_error(eval_series, naive_preds)
naive_rmse = np.sqrt(mean_squared_error(eval_series, naive_preds))
naive_mape = np.mean(np.abs((np.array(eval_series) - np.array(naive_preds)) / np.array(eval_series))) * 100
naive_r2 = r2_score(eval_series, naive_preds)

print("NAIVE BASELINE (persistence)")
print(f"MAE  : {naive_mae:.2f}")
print(f"RMSE : {naive_rmse:.2f}")
print(f"MAPE : {naive_mape:.2f}%")
print(f"R²   : {naive_r2:.4f}")
print("="*40)

# Seasonal-naive baseline (same hour, 24h ago)

seasonal_naive_preds = []
seasonal_naive_actuals = []
seasonal_naive_timestamps = []

for i, ts in enumerate(eval_timestamps):
    abs_idx = WINDOW + i          # this point's index in the full `series`
    lookback_idx = abs_idx - 24
    if lookback_idx < 0:
        continue
    seasonal_naive_preds.append(series[lookback_idx])
    seasonal_naive_actuals.append(eval_series[i])
    seasonal_naive_timestamps.append(ts)

sn_mae = mean_absolute_error(seasonal_naive_actuals, seasonal_naive_preds)
sn_rmse = np.sqrt(mean_squared_error(seasonal_naive_actuals, seasonal_naive_preds))
sn_mape = np.mean(
    np.abs((np.array(seasonal_naive_actuals) - np.array(seasonal_naive_preds))
           / np.array(seasonal_naive_actuals))
) * 100
sn_r2 = r2_score(seasonal_naive_actuals, seasonal_naive_preds)

print("SEASONAL-NAIVE BASELINE (same hour yesterday)")
print(f"MAE  : {sn_mae:.2f}")
print(f"RMSE : {sn_rmse:.2f}")
print(f"MAPE : {sn_mape:.2f}%")
print(f"R²   : {sn_r2:.4f}")
print("="*40)

# Save Results

results = pd.DataFrame({
    "timestamp": eval_timestamps,
    "actual": eval_series,
    "prediction": predictions,
    "naive_prediction": naive_preds
})

results.to_csv("llm_predictions.csv", index=False)
pd.DataFrame(raw_logs).to_csv("llm_raw_logs.csv", index=False)
