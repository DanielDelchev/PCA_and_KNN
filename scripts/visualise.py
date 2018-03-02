import numpy as np
import csv
from PIL import Image
import sys
import os


def visualise(filename, outputDir, labeled = False):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    counter = 0
    starting_index = 0;
    size = 28;
    if labeled:
        starting_index = 1;
    with open(filename) as file:
        reader = csv.reader(file)
        head = next(reader)
        for row in reader:
            data = np.array(row[starting_index:], dtype=np.uint8).reshape(size, size)
            pixel_image = Image.fromarray(data, 'L')
            pixel_image.save(outputDir+'/record'+str(counter)+'.png')
            counter = counter+1


if __name__ == '__main__':

    argc = len(sys.argv)
    if  argc < 3:
        print ("Usage: visualise data.csv outputDir [labeled]")
        exit (0)

    else:
        filename = sys.argv[1];
        outputDir = sys.argv[2];
        labeled = False;
        if (argc==4 and sys.argv[3]=='labeled'):
            labeled = True;
        visualise(filename,outputDir,labeled)