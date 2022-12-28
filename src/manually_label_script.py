import pandas as pd
import glob
import os

LABEL_FILE_PATH = ''
DB_PATH = ''
LAST_PATIENT_FILE = ''

def search_for_keyword(report_path:str, key_word:str) -> bool:
    if os.path.exists(report_path):
        with open(report_path) as f:
            lines = f.readlines()
            for line in lines:
                if line.upper().find(key_word) != -1:
                    return True
    return False


def show_report(report_path:str) -> None:
    if os.path.exists(report_path):
        with open(report_path) as f:
            lines = f.readlines()
            for line in lines:
                print(line)

def get_last_patient_processed()->int:
    with open(LAST_PATIENT_FILE) as f:
        last_patient_processed = int(f.readline().split(':')[-1].strip())
    return last_patient_processed

def save_last_patient_processed(patientId: int)->bool:
    try:
        with open(LAST_PATIENT_FILE, 'w') as f:
            f.write(f"Last_Patient: {patientId}")
        return True
    except:
        return False

def main():
    report_shown = False
    df_labels = pd.read_csv(LABEL_FILE_PATH)
    df_labels['PatientID'] = df_labels['Paciente'].apply(lambda x: int(x.split('-')[-1]))
    labels = dict([(k, []) for k in df_labels.columns])
    
    last_patient_processed = get_last_patient_processed()
    patients = [p.split('/')[-1] for p in glob.glob(os.path.join(DB_PATH, 'Sub*'))]
    patient_ids = [int(p.split('-')[-1]) for p in patients]
    patient_ids.sort()
    
    non_processed_patients = patient_ids[patient_ids.index(last_patient_processed) + 1:]
    
    for patient in non_processed_patients:
        report_paths = glob.glob(os.path.join(DB_PATH, 'Sub-' + str(patient), '*', 'Report', '*.txt'))
        for report in report_paths:
            contains_parkinson = search_for_keyword(report_path=report, key_word='PARKINSON') or \
                                 search_for_keyword(report_path=report, key_word='PÁRKINSON')

            contains_alzheimer = search_for_keyword(report_path=report, key_word='ALZHEIM')

            if contains_alzheimer or contains_parkinson:
                report_shown = True
                print('\n')
                show_report(report)
                print(f"Patient: {'Sub-' + str(patient)}")
                print('\n')
                
                parkinson = input('Has Parkinson ? (Y/N)  ').upper()
                if (parkinson == 'Y'): 
                    parkinson = 1
                elif(parkinson == 'N'):
                    parkinson = 0
                else:
                    raise Exception('Plase type Y or N')

                alzheimer = input('Has Alzheimer? (Y/N)  ').upper()
                if (alzheimer == 'Y'):
                    alzheimer = 1
                elif (alzheimer == 'N'):
                    alzheimer = 0
                else:
                    raise Exception('Plase type Y or N')

                if (parkinson == 1) or (alzheimer == 1):
                    labels['Paciente'].append('Sub-' + str(patient))
                    labels['Parkinson'].append(parkinson)
                    labels['Alzheimer'].append(alzheimer)
                    labels['Estudio'].append('.'.join(report.split('/')[-1].split('.')[:-1]))
                    labels['PatientID'].append(patient)
                else:
                    continue

        
        if report_shown:
            report_shown = False
            out = input('Exit? Y/N  ').upper()
            if (out == 'Y'):
                print(f"Last patient labeled: Sub-{patient}")
                print('\n')
                print('\n')
                break

    df_tmp = pd.DataFrame(labels)
    df_tmp_grp = df_tmp.groupby(['Paciente', 'PatientID', 'Estudio'], as_index=False).sum(['Parkinson', 'Alzheimer'])
    df_tmp_grp = df_tmp_grp[['Paciente', 'Parkinson', 'Alzheimer', 'Estudio', 'PatientID']]
    
    df_labels = df_labels.append(df_tmp_grp, ignore_index=True)
    df_labels.to_csv(LABEL_FILE_PATH, index=False, header=True, sep=',')
    save_result = save_last_patient_processed(patient)

    if save_result:
        print('Last patient successfully saved')
    else:
        print('Problem saving last patient; please copy id from the screen')
    


if __name__ == '__main__':
    main()