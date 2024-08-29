# analyze_results.py
import numpy as np
import pandas as pd

def analyze_model_performance(metrics_history, save_to_csv=False):
    """
    Analyzes and summarizes the model's performance metrics.
    Args:
    - metrics_history: A dictionary containing metric values for each epoch.
    - save_to_csv (bool): Whether to save the analysis results to a CSV file.

    Returns:
    - pd.DataFrame: DataFrame containing summarized performance metrics.
    """
    summary = {
        'Best Accuracy': max(metrics_history['Accuracy']),
        'Best F1 Score': max(metrics_history['F1 Score']),
        'Lowest Loss': min(metrics_history['Loss'])
    }

    # Convert to DataFrame for easy manipulation and export
    df_summary = pd.DataFrame([summary])
    print("Performance Summary:")
    print(df_summary)

    if save_to_csv:
        df_summary.to_csv('model_performance_summary.csv', index=False)

    return df_summary

def compare_models(results_dict):
    """
    Compares the performance of different models based on the provided metrics.
    Args:
    - results_dict: A dictionary containing metrics for different models.

    Returns:
    - None
    """
    comparison_df = pd.DataFrame(results_dict)
    print("Model Comparison:")
    print(comparison_df)
    comparison_df.plot(kind='bar', figsize=(10, 6))
    plt.title('Comparison of Model Performance')
    plt.ylabel('Metric Values')
    plt.show()
