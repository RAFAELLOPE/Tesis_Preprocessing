# Prepare  dataset for T1 images
from nipype import Workflow, Node, Function, MapNode
from nipype.interfaces import fsl
import os
import pandas as pd
from nipype.interfaces.image import Reorient
from sklearn.model_selection import train_test_split
import pandas as pd
from argparse import ArgumentParser
import logging

TMP_path = './data'

def create_workflow(df, output_size, org_base_path):
    
    def resample_image(in_file, out_file, out_size):
        import nibabel.processing
        import nibabel as nib
        vol = nib.load(in_file)
        res = nibabel.processing.conform(vol, out_size)
        nib.save(res, out_file)
        return out_file

    df_tmp = df.copy()
    df_tmp['tmp_path'] = df_tmp['dst_path'].apply(lambda x: os.path.join(TMP_path, x.split('/')[-1]))

    input_files = [os.path.join(org_base_path, p) for p in df_tmp['org_path']]
    resampled_files = [os.path.abspath(p) for p in df_tmp['tmp_path']]
    result_files = [os.path.abspath(p) for p in df_tmp['dst_path']]
    
    resample = MapNode(Function(input_names=["in_file","out_file", "out_size"],
                             output_names=["res_file"],
                             function=resample_image),
                    name='resampling',
                    iterfield=['in_file', 'out_file'])

    resample.inputs.in_file = input_files
    resample.inputs.out_file = resampled_files
    resample.inputs.out_size = output_size

    skullstripping = MapNode(fsl.BET(output_type = 'NIFTI'),
                                     name='skull_strip',
                                     iterfield=['in_file', 'out_file'])  
    skullstripping.inputs.out_file = result_files
    
    # Workflow initialization
    wr_base_dir = os.path.abspath('./data/working_dir')
    wf = Workflow(name='Normalization_Workflow', base_dir=wr_base_dir)

    # First the simple but more restrictive method
    wf.connect(resample, "res_file", skullstripping, "in_file")
    return wf, resampled_files


def run_preprocessing_workflow(df, output_size, org_base_path):
    # Try-catch + log
    df_res = df.copy()
    num_batches = len(set(df['batch']))
    for i in range(num_batches):
        logging.info(f'Starting batch {i}')
        df_b = df_res[df_res['batch'] == i]
        wf, tmp_files = create_workflow(df_b, output_size, org_base_path)
        try:
            wf.run()
        except:
            logging.error(f'Batch {i} failed')
            continue
        finally:
            for f in tmp_files:
                if os.path.exists(f):
                    os.remove(f)


def define_batches_of_data(df, num_batches):
    df_res = df.copy()
    df_res.reset_index(inplace=True, drop=True)
    df_res['batch'] = 0
    N = len(df_res)
    step = N // num_batches
    for j, i in enumerate(range(0, N, step)):
        if N < i + step:
            df_res.loc[i:N, 'batch'] = j
        else:
            df_res.loc[i:i + step, 'batch'] = j
    return df_res


def set_train_val_test_split(df):
    X = df[['org_path', 'HasParkinson']]
    y = df[['HasParkinson']]
    X_train_val, X_test, y_train_val, _ = train_test_split(X, y, 
                                                           random_state=42,
                                                           stratify=y,
                                                           test_size=0.15)
    X_train, X_val, _, _ = train_test_split(X_train_val, 
                                            y_train_val, 
                                            random_state=42,
                                            stratify=y_train_val,
                                            test_size=0.15)
    X_train['dataset'] = 'train'
    X_val['dataset'] = 'validation'
    X_test['dataset'] = 'test'
    X_result = pd.concat([X_train, X_val, X_test])
    return X_result


def create_dest_paths(df, dest_base_path):
    df_tmp = df.copy()
    df_tmp['T1'] = df['NewNiftiPath'].apply(lambda x: x.split('/')[4] == 'T1')
    df_tmp = df_tmp[df_tmp['T1'] == True]
    df_tmp.drop('T1', axis=1, inplace=True)
    def set_dataset_path(x):
        tmp_path = 'NonPD'
        file = x['org_path'].split('/')[-1]
        file = file.replace('.nii.gz', '.nii')
        if x['HasParkinson'] == 1:
            tmp_path = 'PD'
        dst_path = os.path.join(dest_base_path, x['dataset'], tmp_path, file)
        return dst_path
    
    df_tmp['org_path'] = df_tmp['NewNiftiPath'].apply(lambda x: x.replace('.nii.gz', '.nii'))
    df_tmp['org_path'] = df_tmp['org_path'].apply(lambda x: x.replace('neuro_db', 'neuro_db_T1'))
    df_res = set_train_val_test_split(df_tmp)
    df_res['dst_path'] = df_res.apply(set_dataset_path, axis=1)
    return df_res


def manage_arguments():
    parser = ArgumentParser
    parser.add_argument('--db_base_path', 
                        help='Path containing original images', 
                        type=str, 
                        required=False,
                        default='')
    args = parser.parse_args()
    return args



def main(args):
    # Read input variables 
    DB_base_path = args.db_base_path
    assert os.path.exists(DB_base_path) and DB_base_path != ''
    # Configure logging
    
    DB_csv_file = os.path.join(DB_base_path, 'neuro_db_T1', 'neuro_db_metadata_with_labels.csv')
    PD_dataset_path = os.path.join(DB_base_path, 'PD_Dataset')

    logging.basicConfig(filename='./log/results.log',
                        format='%(asctime)s - %(message)s', 
                        datefmt='%d-%b-%y %H:%M:%S', 
                        level=logging.DEBUG)
    # Read csv and put it 
    df_csv = pd.read_csv(DB_csv_file)
    df_tmp = create_dest_paths(df_csv, dest_base_path=PD_dataset_path)
    df_batch = define_batches_of_data(df_tmp, num_batches=9)
    df_batch.to_csv(os.path.join(PD_dataset_path, 'data_index.csv'), index=False)
    run_preprocessing_workflow(df_batch, output_size=(181, 217, 181), org_base_path=DB_base_path)


if __name__ == '__main__':
    args = manage_arguments()
    main(args)
