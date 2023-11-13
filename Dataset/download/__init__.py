import requests
from tqdm import tqdm
import os
import zipfile


def download_file(url, filename=None, verify=True):
    if filename is None:
        filename = url.split('/')[-1]

    # 尝试从文件末尾继续下载（断点续传）
    resume_byte_pos = os.path.getsize(filename) if os.path.exists(filename) else 0

    # 设置请求头，告知服务器我们想从哪个字节开始下载
    headers = {'Range': f'bytes={resume_byte_pos}-'}

    response = requests.get(url, verify=verify, stream=True, headers=headers)
    total_size = int(response.headers.get('content-length', 0)) + resume_byte_pos

    # 创建或继续填充文件
    with open(filename, 'ab') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            initial=resume_byte_pos
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def unzip_file(zip_path, extract_to=None):
    """
    解压ZIP文件到指定目录。

    :param zip_path: ZIP文件的路径。
    :param extract_to: 解压的目标目录。如果为None，则解压到ZIP文件所在的目录。
    """
    if extract_to is None:
        extract_to = os.path.dirname(zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"文件已解压到：{extract_to}")
