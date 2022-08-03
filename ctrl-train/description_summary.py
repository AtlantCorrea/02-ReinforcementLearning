import os
from datetime import datetime

def get_data(root_path, day_path):
    n_models = len(os.listdir(f'trains\{day_path}'))
    desc_day_txt = '-'*10 + '\nDay:\t'+f'{day_path}' +'\nN models:\t'+f'{n_models}' + '\nDescripción:\t'
    desc_modl_txt_a = 'Ø'
    for model_path in os.listdir(f'{root_path}\{day_path}'):
        if 'Description.txt' in os.listdir(f'{root_path}\{day_path}\{model_path}'):
            with open(f'{root_path}\{day_path}\{model_path}\Description.txt') as f:
                desc_modl_txt = f.readline()
            if True or desc_modl_txt != desc_modl_txt_a:

                desc_day_txt += f'\n\tDesde {model_path}: {desc_modl_txt[:75]}...'
            desc_modl_txt_a = desc_modl_txt
    desc_day_txt += '\n'+'-'*0+'\n\n'
    return desc_day_txt





root_path = 'trains'
file_name = 'Summary.txt'
now = datetime.now().strftime('Last update: %D %H\n')

if file_name in os.listdir():
    os.remove(file_name)

with open(file_name, 'a') as f:
    f.write('\t\t'+'Summary'+'\n'*2+f'{now}')
    for day_path in os.listdir(root_path):
        if 'model' in day_path:
            txt_data = get_data(root_path, day_path)
            f.write(txt_data)