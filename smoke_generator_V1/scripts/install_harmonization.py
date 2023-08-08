import subprocess
import shutil
import tarfile
import os
import re 

def modify_dependencies(file_path):
    with open(file_path, "r") as file:
        content = file.read()

    modified_content = re.sub(r'((?:from|import)\s+)scripts(\.\w+)', r'\1harmonization_scripts\2', content)

    with open(file_path, "w") as file:
        file.write(modified_content)

def download_and_rename_harmonization(cwd):
    # Clone the s2am repository
    subprocess.run(['git', 'clone', 'https://github.com/vinthony/s2am'])
    # Move Folder
    directory_path = os.path.join(cwd,'../2harmonization_scripts')
    shutil.move('s2am/scripts',directory_path )
    ## modify imports to make it usable
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            if file_name.endswith(".py"):
                file_path = os.path.join(root, file_name)
                modify_dependencies(file_path)
                print(f"Modified dependencies in {file_path}")
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
    if not(os.path.exists(os.path.join(current_script_path,'../2harmonization_scripts'))) :
        # Download and rename the harmonization repository
        download_and_rename_harmonization(current_script_path)
        # clean github repo
        cleanup()
        print("Harmonization scripts downloaded")
    else :
        print("Harmonization scripts already downloaded")
    """"
    if not(os.path.exists(os.path.join(current_script_path,'../harmonization_scripts/model_path/rascv2.pth.tar'))) :
        # Download and extract the required files
        download_and_extract_files(current_script_path)
        print("Model weights downloaded")
    else :
        print("Models weights already downloaded")
    """