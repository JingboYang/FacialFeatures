import sys, os
import numpy as np
import matplotlib.pyplot as plt

DATA_ROOT = os.path.join(os.environ['HOME'], 'FaceDisguiseDatabase')
DATA_FULL = os.path.join(DATA_ROOT, 'FaceAll')
DATA_TRUTH = os.path.join(DATA_ROOT, 'FaceAll')

#WIDTH,HEIGHT,SEX,SKIN COLOR,MUSTACHE,BEARD,GLASSES,HAT
truth_index = {'name': 0, 'w': 1, 'h': 2, 'sex': 3, 'skin': 4, 'mus': 5, 'bea': 6, 'gls': 7, 'hat': 8}

def main():
    
    truth = []
    for f in os.listdir(DATA_TRUTH):
        truth_path = os.path.join(DATA_TRUTH, f)

        # sub0023.txt,950,634,1,0,0,0,0,1
        with open(truth_path, 'r') as f:
            line = f.readlines()[0]
            splits = line.split(',')
        
        truth.append(splits)

    
    


if __name__ == '__main__':
    main()