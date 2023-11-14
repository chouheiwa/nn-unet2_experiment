from skimage import io
import multiprocessing
import shutil

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles

from ImagePrepare import generate_dataset_json
from path import get_generated_dataset_path, get_original_dataset_path


def load_and_convert_case(input_image: str, input_seg: str, output_image: str, output_seg: str):
    seg = io.imread(input_seg)
    seg[seg < 155] = 0
    seg[seg > 155] = 1
    io.imsave(output_seg, seg, check_contrast=False)
    shutil.copy(input_image, output_image)


def main_func():
    # extracted archive from https://www.kaggle.com/datasets/insaff/massachusetts-roads-dataset?resource=download
    base_dataset_name = 'Dataset001'

    source = get_original_dataset_path(base_dataset_name)

    generated_path = get_generated_dataset_path(base_dataset_name)

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
            r.append(
                p.starmap_async(
                    load_and_convert_case,
                    ((
                         join(train_source, 'Rail surface images', v),
                         join(train_source, 'GroundTruth', v),
                         join(imagestr, v[:-4] + '_0000.png'),
                         join(labelstr, v[:-4] + '.png')
                     ),)
                )
            )
        _ = [i.get() for i in r]

    generate_dataset_json(generated_path, {0: 'Grey'}, {'background': 0, 'bug': 1},
                          num_train, '.png', dataset_name=generated_path)


if __name__ == '__main__':
    main_func()
