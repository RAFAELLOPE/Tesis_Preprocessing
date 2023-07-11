import os
from argparse import ArgumentParser
import pandas as pd
import shutil
from pathlib import Path

def move_data(org:str, dst:str) -> None:
    if os.path.exists(org):
        if os.path.exists(dst) == False:
            create_destination_folder(dst)
        for file_name in os.listdir(org):
        # construct full file path
            source = org + '/' +file_name
            destination = dst + '/' +file_name
            # move only files
            if os.path.isfile(source):
                shutil.move(source, destination)
                print('Moved:', file_name)


def get_metadata(path:str)->pd.DataFrame:
    df = pd.read_csv(path, sep=',', index_col=False)
    return df

def create_destination_folder(path:str)->None:
    Path(path).mkdir(parents=True, exist_ok=True)
    return

def main(args:ArgumentParser)-> None:
    def create_destination_path(x):
        v = x['org'].split('/')
        if x['HasParkinson'] == 1:
            r = 'neuro_db_pd/PD/' + '/'.join(v[1:4])
        else:
            r = 'neuro_db_pd/Control/' + '/'.join(v[1:4])
        return r

    df = get_metadata(args.csv_path)
    df = df[['NewNiftiPath', 'HasParkinson']]
    df['org'] = df['NewNiftiPath'].apply(lambda x:'neuro_db_T1/' +  '/'.join(x.split('/')[1:-1]))
    df['dst'] = df.apply(create_destination_path, axis=1)
    for index, row in df.iterrows():
        move_data(row['org'], row['dst'])

if __name__== "__main__":
    parser = ArgumentParser()
    parser.add_argument('--csv_path',
                        type=str,
                        required=False,
                        default='neuro_db_metadata_with_labels.csv')
    args = parser.parse_args()
    main(args)