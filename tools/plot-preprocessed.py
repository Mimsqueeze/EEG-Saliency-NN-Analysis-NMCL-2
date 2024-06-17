import matplotlib.pyplot as plt
import pandas as pd
import os
# ----------------------- author: Minsi Hu ---------------------------

# ----------------------- program description: -----------------------
#               write what this source file does, briefly.
# --------------------------------------------------------------------

# -- write a brief description of your main function --
def main():
    # Subject number
    n = 1
            
    # Level map
    l_map = {"easy":"Easy", "med":"Medium", "diff":"Difficult"}

    # Extract data from sessions 1, 2, 3 and levels easy, med, diff
    for s in [1]:
        for l in ["easy", "med", "diff"]:
            # Read in the data saved into csv
            df = pd.read_csv(f"./data/sub-{n:02}/ses-S{s}/{l}.csv")

            # Drop the extra index column
            df = df.drop("Unnamed: 0", axis=1)

            # Get 0.50 seconds of data
            data = df.iloc[1000:1250+1]
            print(data.columns)
            data = data[["FCz", "FT7", "FT8", "AFz", "AF7", "AF8", "Pz", "P8", "P7"]]
            data.index -= 1000
            data.index *= 2
            data.index /= 1000
            print(data)

            # Plot the data
            ax = data.plot(kind="line", legend=False)
            ax.set_xlim(0, 0.50)
            ax.set_ylim(-35, 35)

            plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=8)
            plt.yticks([-30, -20, -10, 0, 10, 20, 30], fontsize=8)
            plt.tick_params(axis="both", length=2, pad=1)

            plt.title(f"Preprocessed EEG Data MATB-{l_map[l]}")
            plt.xlabel("Time (s)", labelpad=3)
            plt.ylabel("Voltage (mV)", labelpad=1)
            plt.tight_layout()

            # Make the directory if the directory does not exist
            target_directory = f"./plots/sub-{n:02}/ses-S{s}/preprocessed/"
            if not os.path.exists(target_directory): os.makedirs(target_directory)

            print("Saved plot!")

            plt.savefig(f"{target_directory}/{l}-raw.svg")
            plt.clf()
            plt.close()
    
if __name__ == '__main__':
    main()