# fairness_evaluation.py
from sklearn.metrics import mean_absolute_error
import pandas as pd
import torch

def evaluate_fairness_across_groups(model, test_loader, ethnicities_array):
    """
    Evaluate model performance across demographic groups.

    Args:
        model: trained multimodal model
        test_loader: DataLoader yielding (images, metadata, targets)
        ethnicities_array: list/array of ethnicities corresponding to each sample

    Returns:
        fairness_report: dict with MAE per ethnic group
    """
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for images, metadata, targets in test_loader:
            images, metadata, targets = images.to(model.device), metadata.to(model.device), targets.to(model.device)
            outputs = model(images, metadata)
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(targets.cpu().numpy().flatten())

    results_df = pd.DataFrame({
        'predicted': predictions,
        'actual': actuals,
        'ethnicity': ethnicities_array
    })

    # Calculate MAE per ethnic group
    fairness_report = {}
    for ethnicity in results_df['ethnicity'].unique():
        mask = results_df['ethnicity'] == ethnicity
        group_mae = mean_absolute_error(results_df.loc[mask, 'actual'], results_df.loc[mask, 'predicted'])
        fairness_report[ethnicity] = {
            'mae': group_mae,
            'count': mask.sum()
        }

    return fairness_report

# ethnicities_array = meta_df.loc[test_dataset.indices, 'ethnicity'].values
# report = evaluate_fairness_across_groups(model, test_loader, ethnicities_array)
# print(report)

