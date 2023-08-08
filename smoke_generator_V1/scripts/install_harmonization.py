import subprocess
import shutil
import tarfile
import os

def download_and_rename_harmonization(cwd):
    # Clone the s2am repository
    subprocess.run(['git', 'clone', 'https://github.com/vinthony/s2am'])
    # Move Folder
    shutil.move('s2am/scripts', os.path.join(cwd,'../harmonization_scripts'))

    # Rename the s2am repository to harmonization

def download_and_extract_files(cwd):
    # Download the required files
    os.system('gdown https://drive.google.com/uc?id=1bm1ZdZ4xmV9fKCQBDsulvYwrxPAidZ3T')
    # Move Folder
    os.mkdir(os.path.join(cwd, '../harmonization_scripts/model_path/'))
    shutil.move('rascv2.pth.tar', os.path.join(cwd,'../harmonization_scripts/model_path/rascv2.pth.tar'))

def cleanup():
    # Clean up the downloaded harmonization directory
    subprocess.run(['rmdir', '/s', '/q', 's2am'], shell=True)
    
if __name__ == '__main__':
    current_script_path = os.path.abspath(__file__)
    print(current_script_path)
    # Download and rename the harmonization repository
    if not(os.path.exists(os.path.join(current_script_path,'../harmonization_scripts'))) :
        # Download and rename the harmonization repository
        download_and_rename_harmonization(current_script_path)
        # clean github repo
        cleanup()
        print("Harmonization scripts downloaded")
    else :
        print("Harmonization scripts already downloaded")
    if not(os.path.exists(os.path.join(current_script_path,'../harmonization_scripts/model_path/rascv2.pth.tar'))) :
        # Download and extract the required files
        download_and_extract_files(current_script_path)
        print("Model weights downloaded")
    else :
        print("Models weights already downloaded")
