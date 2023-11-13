import os

from Dataset.download import download_file, unzip_file

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

down_load_url = 'https://scholar.cu.edu.eg/Dataset_BUSI.zip'
down_load_file_name = 'Dataset_BUSI.zip'

down_load_file_path = os.path.join(current_dir, down_load_file_name)

unzip_folder_name = 'Dataset_BUSI_with_GT'

if __name__ == '__main__':
    download_file(down_load_url, down_load_file_path, verify=False)
    unzip_file(down_load_file_path)

