# Imports
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------- author: Minsi Hu -----------------------

# ----------------------- program description: -----------------------
#            Plots the performance of different classifiers
#              in classifying mental workload of EEG data
# --------------------------------------------------------------------

"""
The main() function

Reads in the results data and plots the performance of different classifiers
based on different metrics
"""
def main():
    for tup in [("2", "binary"), ("3", "ternary")]:
      # Create directory if it doesn't exist already
      dir = f"./plots/results/{tup[1]}/"
      if not os.path.exists(dir): os.makedirs(dir)

      print("Plotting accuracy...")

      # Read the results data
      df1 = pd.read_csv(f"./results/{tup[1]}/MLP_{tup[0]}-coh-only.csv", engine="python", 
                      header=0, index_col=0)
      df2 = pd.read_csv(f"./results/{tup[1]}/CNN_{tup[0]}-coh-only.csv", engine="python", 
                      header=0, index_col=0)

      # Merge the results into a single dataframe
      data = df1.merge(df2, left_index=True, right_index=True)
      data = data.apply(pd.to_numeric, errors='coerce')
      data = data.round(4)
      data.drop(index=["Summary", "Number of Epochs", "Mean Fit Time"], inplace=True)

      print(data)
      fig, ax = plt.subplots(figsize=(3, 6))
      ax.plot(data.columns, data.loc["Mean Train Accuracy"], marker='^', markersize=6, label="Train")
      ax.plot(data.columns, data.loc["Mean Test Accuracy"], marker='s', markersize=6, label="Test")

      ax.legend(fontsize=8, frameon=False, loc=(0.65, 0.75))
      plt.xlabel("classifier")
      plt.ylabel("accuracy")
      plt.xticks([0,1], data.columns, fontsize=8)
      plt.yticks(fontsize=8)
      plt.xlim(-0.5, 1.5)
      if tup[1] == "binary":
        plt.ylim(0.97, 1)
      else:
        plt.ylim(0.85, 1)
      plt.tick_params(top=True, right=True, axis="both", length=3, pad=2, direction='in')
      plt.savefig(f"{dir}accuracy.svg", dpi=1000, bbox_inches="tight", format="svg")
      plt.clf()
      plt.close()

      print("Plotting loss...")
      fig, ax = plt.subplots(figsize=(3, 6))
      ax.plot(data.columns, data.loc["Mean Train Loss"], marker='^', markersize=6, label="Train")
      ax.plot(data.columns, data.loc["Mean Test Loss"], marker='s', markersize=6, label="Test")

      ax.legend(fontsize=8, frameon=False, loc=(0.65, 0.75))
      plt.xlabel("classifier")
      plt.ylabel("loss")
      plt.xticks([0,1], data.columns, fontsize=8)
      plt.yticks(fontsize=8)
      plt.xlim(-0.5, 1.5)
      if tup[1] == "binary":
        plt.ylim(0, 0.16)
      else:
        plt.ylim(0, 0.7)
      plt.tick_params(top=True, right=True, axis="both", length=3, pad=2, direction='in')
      plt.savefig(f"{dir}loss.svg", dpi=1000, bbox_inches="tight", format="svg")
      plt.clf()
      plt.close()

if __name__ == '__main__':
  main()