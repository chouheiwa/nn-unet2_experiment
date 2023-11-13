import os


def get_root_path():
    # 获取当前py文件的绝对路径
    cur_path = os.path.dirname(os.path.realpath(__file__))
    # 获取父目录
    root_path = os.path.dirname(cur_path)
    return root_path


def join_path(left: str, right: str):
    if right is None:
        return left
    return os.path.join(left, right)


def get_dataset_path(name: str = None):
    return join_path(os.path.join(get_root_path(), 'Dataset'), name)


def get_original_dataset_path(generated_name: str = None):
    return join_path(get_dataset_path('original'), generated_name)


def get_generated_dataset_path(generated_name: str = None):
    return join_path(get_dataset_path('generated'), generated_name)


def get_preprocess_path(generated_name: str = None):
    return join_path(get_dataset_path('preprocess'), generated_name)


def get_result_path(generated_name: str = None):
    return join_path(get_dataset_path('result'), generated_name)
