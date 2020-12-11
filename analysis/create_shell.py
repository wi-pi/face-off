import os

ROOT = os.getcwd()
ROOT = os.path.join(ROOT, 'csv')

folders = ['li', 'l2_fine', 'l2_coarse', 'pgd']


for folder in folders:
    print("#"+folder)
    path = os.path.join(ROOT, folder)
    files = os.listdir(path)
    for f in files:
        print("python stats.py "+f+" "+folder)
    print("\n")
