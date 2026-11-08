import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plotPredictions():
    # Load data
    data = pd.read_csv("predictions.csv")
    predictions = data["prediction"]
    actual = data["actual"]

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(actual, predictions, s=12, alpha=0.35)

    bounds = [np.min([ax.get_xlim(), ax.get_ylim()]),
                    np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(bounds, bounds)
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.grid(visible=True, axis='both')
    ax.axhline(0, color="black")  # Horizontal axis at y=0
    ax.axvline(0, color="black")  # Vertical axis at x=0
    ax.set_xlabel("Actual +/-")
    ax.set_ylabel("Predicted +/-")
    ax.set_title("Actual vs Predicted +/-")
    fig.savefig("predictions.png")

def plotImportances():
    # Load data
    importances = pd.read_csv("importances.csv")
    importances = importances.sort_values(by=["importance"], ascending=False)

    # Plot
    fig, ax = plt.subplots()
    top = importances.head(15)
    ax.barh(y=top["feature"], 
            width=top["importance"])
    ax.set_xlabel("Feature importance (share of total)")
    ax.set_ylabel("Top 15 Features by importance")
    ax.set_title("Feature Importances")
    fig.savefig("importances.png", bbox_inches="tight")

def plotError():
    # Load data
    data = pd.read_csv("predictions.csv")
    predictions = data["prediction"]
    actual = data["actual"]

    # Calculate metrics
    error = actual.sub(predictions)
    mae = error.abs().mean()
    rmse = np.sqrt((error ** 2).mean())

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(predictions, error, s=12, alpha=0.35)
    ax.axhline(0, color="black")
    ax.set_xlabel("Predicted +/-")
    ax.set_ylabel("Residuals (actual - predicted)")
    ax.set_title("Residual Plot")
    ax.text(x=0.80, 
             y=0.90, 
             s=f"MAE={mae:.2f}\nRMSE={rmse:.2f}", 
             transform=ax.transAxes)
    fig.savefig("residuals.png")

def calibrationPlot():
    # Load data
    data = pd.read_csv("predictions.csv")
    data["win"] = data["actual"] > 0

    # Create bins and summarize win probs
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 
            0.6, 0.7, 0.8, 0.9, 1.0]
    data["bin"] = pd.cut(data["win_prob"], bins=bins)
    summary = data.groupby("bin", observed=True).agg(
        avg_pred=("win_prob", "mean"),
        win_rate=("win", "mean"),
        count=("GAME_ID", "count"),
    )

    # Filter out bins with less than 10 games to not skew results
    summary = summary[summary["count"] >= 10]
    # Plot
    fig, ax = plt.subplots()
    ax.plot(summary["avg_pred"], summary["win_rate"], "o-")
    ax.plot([0, 1], [0, 1], "--")
    for _, row in summary.iterrows():
        ax.annotate(
            text=f"n={int(row['count'])}",
            xy=(row["avg_pred"], row["win_rate"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9
        )
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Actual win rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Calibration Plot")
    fig.savefig("calibration.png", bbox_inches="tight")

# plotPredictions()
# plotError()
# plotImportances()
# calibrationPlot()