import pyjson5 as json
import subprocess
import os
from pathlib import Path
import datasets.ReforesTreeDataset as ReforesTreeDataset
import datasets.SSL4EOLDataset as SSL4EOL
import datasets.SSL4EOS1S2Dataset as SSL4EOS1S2


if __name__=='__main__':
    configs = json.load(open('configs/download/download.json','r'))
    dataset = configs['dataset'].lower()
    root_path = configs['root_path']
    
    if dataset=='reforestree':
        root_path = os.path.join(root_path,'Reforestree')
        reforestree_configs = {"root_path":root_path,"download":True,"checksum":True,"augment":False,"normalization":"none"}
        ReforesTreeDataset.ReforesTreeDataset(reforestree_configs)
        exit(0)
    elif dataset=='ssl4eol':
        ssl4eol_configs = {"root_path":root_path,"download":True,"checksum":True,"augment":False,"normalization":"none","split":"oli_tirs_toa","seasons":4}
        SSL4EOL.SSL4EOL(ssl4eol_configs)
        exit(0)

    download_script = Path('downloading_scripts/' + dataset + '.sh')

    if not download_script.is_file():
        print('Dataset is not supported for downloading!')
        exit(2)
    
    process = subprocess.Popen([str(download_script) + " " + root_path], shell=True, stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline().decode()
        if output == "" and process.poll() is not None:
            break
        print(output.strip())

    process.wait()

    # Check if the download was successful
    if process.returncode == 0:
        print("Process finished successfully")
    else:
        print("Process failed!")


