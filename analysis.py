import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plotPredictions():
    data = pd.read_csv("predictions.csv")
    predictions = data["prediction"]
    actual = data["actual"]

    fig, ax = plt.subplots()
    ax.scatter(actual, predictions, s=12, alpha=0.35)

    bounds = [np.min([ax.get_xlim(), ax.get_ylim()]),
                    np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(bounds, bounds)
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.grid(visible=True, axis='both')
    # ax.axes()
    ax.axhline(0, color="black")  # Horizontal axis at y=0
    ax.axvline(0, color="black")  # Vertical axis at x=0
    plt.xlabel("Actual +/-")
    plt.ylabel("Predicted +/-")
    plt.title("Actual vs Predicted +/-")
    # plt.show()
    fig.savefig("predictions.png")

def plotImportances():
    importances = pd.read_csv("importances.csv")
    importances = importances.sort_values(by=["importance"], ascending=False)
    # print(importances)
    fig, ax = plt.subplots()
    top = importances.head(15)
    ax.barh(y=top["feature"], 
            width=top["importance"])
    # plt.show()
    plt.xlabel("Feature importance (share of total)")
    plt.ylabel("Top 15 Features by importance")
    plt.title("Feature Importances")
    fig.savefig("importances.png", bbox_inches="tight")

def plotError():
    data = pd.read_csv("predictions.csv")
    predictions = data["prediction"]
    actual = data["actual"]

    error = actual.sub(predictions)
    mae = error.abs().mean()
    rmse = np.sqrt((error ** 2).mean())

    fig, ax = plt.subplots()
    ax.scatter(predictions, error, s=12, alpha=0.35)
    ax.axhline(0, color="black")
    plt.xlabel("Predicted +/-")
    plt.ylabel("Residuals (actual - predicted)")
    plt.title("Residual Plot")
    plt.text(x=0.80, 
             y=0.90, 
             s=f"MAE={mae:.2f}\nRMSE={rmse:.2f}", 
             transform=ax.transAxes)
    fig.savefig("residuals.png")

# plotError()