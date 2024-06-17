import subprocess
import multiprocessing

# ----------------------- author: Minsi Hu ---------------------------

# ----------------------- program description: -----------------------
#              A utility script for preprocessing the data
#                  for all 29 subjects concurrently
# --------------------------------------------------------------------

# How many subjects to process at a time
M = 3

def run_script(n, l):
    subprocess.run(["python", "-u", ".\src\\feature-extraction.py", str(n), l])

# Concerrently preprocesses the data for all 29 subjects
def main():

    # Subject number
    n = 1

    # Process subjects from k to k + m at a time
    for k in range(1, 30, M):
            
        # List of processes
        p_list = []

        # Begin processes
        while n < k + M and n < 30:
            for l in ["easy", "med", "diff"]:
                p = multiprocessing.Process(target=run_script, args=(n,l,))
                p_list += [p]
                p.start()
            n += 1
        
        # Join processes
        for p in p_list:
            p.join()
            
    print("Finished feature extraction!")

if __name__ == '__main__':
    main()
