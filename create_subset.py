import pandas as pd
import subprocess
import os
from tqdm import tqdm
table = pd.read_csv('data/slurp/hg_face_data/data/train/metadata.csv')
import os 

for i in tqdm(range(len(table['text']))):
    file_name = table['file_name'][i]
    commend = f'cp data/slurp/hg_face_data/data/train/{file_name} data/slurp/hg_face_data/data/train_subset/{file_name}'
    subprocess.run(commend, shell=True)