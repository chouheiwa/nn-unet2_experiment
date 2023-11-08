import multiprocessing
import shutil
from multiprocessing import Pool

from batchgenerators.utilities.file_and_folder_operations import *

from generate_dataset_json import generate_dataset_json
from skimage import io
from acvl_utils.morphology.morphology_helper import generic_filter_components
from scipy.ndimage import binary_fill_holes


def load_and_covnert_case(input_image: str, input_seg: str, output_image: str, output_seg: str,
                          min_component_size: int = 50):
    seg = io.imread(input_seg)
    seg[seg < 155] = 0
    seg[seg > 155] = 1
    image = io.imread(input_image)
    # image = image.sum(2)
    mask = image == (3 * 255)
    # the dataset has large white areas in which road segmentations can exist but no image information is available.
    # Remove the road label in these areas
    mask = generic_filter_components(mask, filter_fn=lambda ids, sizes: [i for j, i in enumerate(ids) if
                                                                         sizes[j] > min_component_size])
    mask = binary_fill_holes(mask)
    seg[mask] = 0
    io.imsave(output_seg, seg, check_contrast=False)
    shutil.copy(input_image, output_image)


def load_and_covnert_case1():
    input_seg = '/Users/chouheiwa/Desktop/实验/Dataset/original/Type-I RSDDs dataset/GroundTruth/rail_1.jpg'
    input_image = '/Users/chouheiwa/Desktop/实验/Dataset/original/Type-I RSDDs dataset/Rail surface images/rail_1.jpg'
    output_dir = '/Users/chouheiwa/Desktop/数据对比'
    seg = io.imread(input_seg)
    seg[seg == 255] = 1
    image = io.imread(input_image)
    for index in range(1, 200):
        # image = image.sum(2)
        new_seg = seg.copy()
        mask = image == (3 * 255)
        # the dataset has large white areas in which road segmentations can exist but no image information is available.
        # Remove the road label in these areas
        mask = generic_filter_components(mask, filter_fn=lambda ids, sizes: [i for j, i in enumerate(ids) if
                                                                             sizes[j] > index])
        mask = binary_fill_holes(mask)
        new_seg[mask] = 0
        io.imsave(join(output_dir, f'{index:03d}.png'), new_seg, check_contrast=False)
    # shutil.copy(input_image, output_image)

def get_root_path():
    # 获取当前py文件的绝对路径
    cur_path = os.path.dirname(os.path.realpath(__file__))
    # 获取父目录
    root_path = os.path.dirname(cur_path)
    return root_path

def get_generated_dataset_path(generated_name):
    dataset_path = os.path.join(get_root_path(), 'Dataset', 'generated')
    return os.path.join(dataset_path, generated_name)


def get_origin_dataset_path(original_name):
    # 获取父目录的父目录
    dataset_path = os.path.join(get_root_path(), 'Dataset', 'original')
    return os.path.join(dataset_path, original_name)


def main_func():
    # extracted archive from https://www.kaggle.com/datasets/insaff/massachusetts-roads-dataset?resource=download
    source = get_origin_dataset_path('Type-I RSDDs dataset')

    ground_truth = join(source, 'GroundTruth')
    images = join(source, 'Rail surface images')

    generate_dataset_name = 'Dataset001'

    generated_path = get_generated_dataset_path(generate_dataset_name)

    imagestr = join(generated_path, 'imagesTr')
    imagests = join(generated_path, 'imagesTs')
    labelstr = join(generated_path, 'labelsTr')
    labelsts = join(generated_path, 'labelsTs')
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    train_source = source

    # test_source = join(source, 'testing')

    with multiprocessing.get_context("spawn").Pool(8) as p:
        # not all training images have a segmentation
        data_path = join(train_source, 'Rail surface images')
        valid_ids = subfiles(data_path, join=False, suffix='jpg')
        num_train = len(valid_ids)
        r = []
        for v in valid_ids:
            # load_and_covnert_case(
            #     join(train_source, 'Rail surface images', v),
            #     join(train_source, 'GroundTruth', v),
            #     join(imagestr, v[:-4] + '_0000.png'),
            #     join(labelstr, v[:-4] + '.png'),
            #     100
            # )
            r.append(
                p.starmap_async(
                    load_and_covnert_case,
                    ((
                         join(train_source, 'Rail surface images', v),
                         join(train_source, 'GroundTruth', v),
                         join(imagestr, v[:-4] + '_0000.png'),
                         join(labelstr, v[:-4] + '.png'),
                         50
                     ),)
                )
            )

        # test set
        # valid_ids = subfiles(join(test_source, 'output'), join=False, suffix='png')
        # for v in valid_ids:
        #     r.append(
        #         p.starmap_async(
        #             load_and_covnert_case,
        #             ((
        #                  join(test_source, 'input', v),
        #                  join(test_source, 'output', v),
        #                  join(imagests, v[:-4] + '_0000.png'),
        #                  join(labelsts, v),
        #                  50
        #              ),)
        #         )
        #     )
        _ = [i.get() for i in r]

    generate_dataset_json(generated_path, {0: 'Grey'}, {'background': 0, 'bug': 255},
                          num_train, '.png', dataset_name=generated_path)


if __name__ == "__main__":
    main_func()
