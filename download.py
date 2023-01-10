import os
import shutil
import random
import string
import pandas as pd
from tqdm import tqdm 
import gdown 

def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

if __name__ == '__main__':
    df = pd.read_csv('./data/link_report_1912.csv')
    for url in tqdm(df.link.tolist()):
        try:
            name = get_random_string(10)
            output = f"./data/unlabel_data/{name}.jpg"
            gdown.download(url, output, quiet=True, verify=False)
        except:
            print(url)