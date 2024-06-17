import subprocess
import multiprocessing

# ----------------------- author: Minsi Hu ---------------------------

# ----------------------- program description: -----------------------
#              A utility script for preprocessing the data
#                  for all 29 subjects concurrently
# --------------------------------------------------------------------

def run_script(n):
    subprocess.run(["python", "-u", ".\src\preprocessing.py", str(n)])

# Concerrently preprocesses the data for all 29 subjects
def main():

    # List of processes
    p_list = []
    for n in range(1, 30):
        p = multiprocessing.Process(target=run_script, args=(n,))
        p_list += [p]
        p.start()
    
    for p in p_list:
        p.join()
    

if __name__ == '__main__':
    main()
