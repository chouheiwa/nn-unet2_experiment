import multiprocessing
import os

from skimage import io

from path import get_original_dataset_path, get_result_path
from batchgenerators.utilities.file_and_folder_operations import *

from skimage import io
import numpy as np


def overlay_images_with_transparency(
        original_path,
        ground_truth_path,
        inference_path,
        save_path
):
    # 读取图像
    original = io.imread(original_path)
    ground_truth = io.imread(ground_truth_path)
    inference = io.imread(inference_path)

    # 确保图像尺寸一致
    if original.shape[:2] != ground_truth.shape[:2] or original.shape[:2] != inference.shape[:2]:
        raise ValueError("All images must have the same dimensions")

    # 将灰度图转换为RGB
    if len(original.shape) == 2:
        original = np.stack((original,) * 3, axis=-1)

    # 创建输出图像
    output = original.astype(np.float32)

    # 定义蒙层颜色和透明度
    red = np.array([255, 0, 0], dtype=np.float32)
    green = np.array([0, 255, 0], dtype=np.float32)
    yellow = np.array([255, 255, 0], dtype=np.float32)
    alpha = 0.4  # 透明度

    # 错误标注（红色蒙层）
    incorrect_mask = (inference == 1) & (ground_truth < 155)
    output[incorrect_mask] = alpha * red + (1 - alpha) * output[incorrect_mask]

    # 正确标注（绿色蒙层）
    correct_mask = (inference == 1) & (ground_truth >= 155)
    output[correct_mask] = alpha * green + (1 - alpha) * output[correct_mask]

    # 未标注区域（黄色蒙层）
    missed_mask = (inference != 1) & (ground_truth >= 155)
    output[missed_mask] = alpha * yellow + (1 - alpha) * output[missed_mask]

    io.imsave(save_path, np.clip(output, 0, 255).astype(np.uint8))


def get_name(original_name, suffix):
    datas = original_name.split('.')
    datas[1] = suffix
    return '.'.join(datas)


if __name__ == '__main__':
    result_path = get_result_path('nnsam')
    result_path = join(result_path, 'Dataset001', 'nnUNetTrainer__nnUNetPlans__2d', 'fold_4', 'validation')

    source = join(get_original_dataset_path(), 'Type-I RSDDs dataset')

    ground_truth = join(source, 'GroundTruth')
    images = join(source, 'Rail surface images')

    inference_result = join(source, 'inference_result')

    maybe_mkdir_p(inference_result)

    valid_id = subfiles(result_path, join=False, suffix='png')

    with multiprocessing.get_context("spawn").Pool(8) as p:
        r = []
        for i in valid_id:
            r.append(
                p.starmap_async(
                    overlay_images_with_transparency,
                    (
                        (
                            join(images, get_name(i, 'jpg')),
                            join(ground_truth, get_name(i, 'jpg')),
                            join(result_path, i),
                            join(inference_result, i)
                        ),
                    )
                )
            )
        _ = [i.get() for i in r]
