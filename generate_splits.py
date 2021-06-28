import os
import sys
import csv
import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN = 0.98

if __name__ == '__main__':

    input_dir = sys.argv[1]
    out_filename = sys.argv[2]
    dataset_name = sys.argv[3]
    database_path = sys.argv[4]
    images = os.listdir(input_dir)

    # Generate train/test splits
    train, test = train_test_split(images, test_size=(1-TRAIN))

    with open(out_filename, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(['filename', 'id', 'split', 'dataset'])
        for i, img in enumerate(images):
            if img in train:
                tsv_writer.writerow([img, i, 'train', dataset_name])
            if img in test:
                tsv_writer.writerow([img, i, 'test', dataset_name])
