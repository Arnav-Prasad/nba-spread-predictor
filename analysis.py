import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("predictions.csv")
predictions = data["prediction"]
actual = data["actual"]

fig_s, ax_s = plt.subplots()
ax_s.scatter(actual, predictions, s=12, alpha=0.35)

bounds = [np.min([ax_s.get_xlim(), ax_s.get_ylim()]),
                np.max([ax_s.get_xlim(), ax_s.get_ylim()])]
ax_s.plot(bounds, bounds)
ax_s.set_xlim(bounds)
ax_s.set_ylim(bounds)
ax_s.grid(visible=True, axis='both')
# ax.axes()
ax_s.axhline(0, color="black", linewidth=1.2)  # Horizontal axis at y=0
ax_s.axvline(0, color="black", linewidth=1.2)  # Vertical axis at x=0
plt.xlabel("Actual +/-")
plt.ylabel("Predicted +/-")
plt.title("Actual vs Predicted +/-")
# plt.show()
fig_s.savefig("scatter.png")

importances = pd.read_csv("importances.csv")
importances = importances.sort_values(by=["importance"], ascending=False)
# print(importances)
fig_b, ax_b = plt.subplots()
top = importances.head(15)
ax_b.barh(y=top["feature"], 
          width=top["importance"])
# plt.show()
plt.xlabel("Feature importance (share of total)")
plt.title("Feature Importances")
fig_b.savefig("bar.png", bbox_inches="tight")
