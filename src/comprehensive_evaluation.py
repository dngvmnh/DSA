# comprehensive_evaluation.py
def comprehensive_model_evaluation():
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Load all CSVs
    labels_df = pd.read_csv("labels.csv")
    baseline_df = pd.read_csv("baseline_predictions.csv")
    cnn_df = pd.read_csv("cnn_predictions.csv")
    multimodal_df = pd.read_csv("multimodal_predictions.csv")

    # Normalize keys: add .heic to labels
    labels_df['filename'] = labels_df['image_id'].astype(str) + '.heic'
    labels_df.rename(columns={'hgb': 'hgb_value'}, inplace=True)

    # Merge consistently on filename
    results = labels_df[['filename', 'hgb_value']].copy()
    results = results.merge(baseline_df, on='filename', how='left')
    results = results.merge(cnn_df, on='filename', how='left')
    results = results.merge(multimodal_df, on='filename', how='left')

    # Drop rows with any missing predictions
    results.dropna(subset=['predicted_hgb_baseline', 'predicted_hgb', 'predicted_hgb_multi'], inplace=True)

    # Evaluate
    models_performance = {}
    for model_name, pred_col in [
        ("Baseline", "predicted_hgb_baseline"),
        ("CNN", "predicted_hgb"),
        ("Multimodal", "predicted_hgb_multi")
    ]:
        mae = mean_absolute_error(results['hgb_value'], results[pred_col])
        rmse = np.sqrt(mean_squared_error(results['hgb_value'], results[pred_col]))
        mean_bias = np.mean(results[pred_col] - results['hgb_value'])
        r2 = r2_score(results['hgb_value'], results[pred_col])

        models_performance[model_name] = {
            'MAE': mae,
            'RMSE': rmse,
            'Mean_Bias': mean_bias,
            'R2': r2,
            'Meets_Target': "✓" if mae <= 0.8 else "✗"
        }

    # Print results
    perf_df = pd.DataFrame(models_performance).T
    print("=== Model Performance Comparison ===")
    print(perf_df.round(3))

    best_model = min(models_performance, key=lambda m: models_performance[m]['MAE'])
    print(f"\nBest Model: {best_model} (MAE: {models_performance[best_model]['MAE']:.3f})")

    return best_model, perf_df

if __name__ == "__main__":
    best_model, performance_table = comprehensive_model_evaluation()
