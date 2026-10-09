import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
ax.axhline(0, color="black", linewidth=1.2)  # Horizontal axis at y=0
ax.axvline(0, color="black", linewidth=1.2)  # Vertical axis at x=0
plt.xlabel("Actual +/-")
plt.ylabel("Predicted +/-")
plt.title("Actual vs Predicted +/-")
plt.show()
fig.savefig("scatter.png")