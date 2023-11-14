import os

import torch
from path import get_generated_dataset_path, get_preprocess_path, get_result_path


def setup_env():
    os.environ['MODEL_NAME'] = 'nnsam'
    os.environ['nnUNet_raw'] = get_generated_dataset_path()
    os.environ['nnUNet_preprocessed'] = get_preprocess_path()
    os.environ['nnUNet_results'] = get_result_path()
    os.environ['nnUNet_visual_port'] = '6666'


def pre_process_database():
    dataset_ids = [1]  # -d 参数
    verify_dataset_integrity = True  # --verify_dataset_integrity 参数
    fpe = 'DatasetFingerprintExtractor'  # -fpe 参数的默认值
    npfp = 8  # -npfp 参数的默认值
    clean = False  # --clean 参数的默认值
    verbose = False  # --verbose 参数的默认值
    pl = 'ExperimentPlanner'  # -pl 参数的默认值
    gpu_memory_target = 8  # -gpu_memory_target 参数的默认值
    preprocessor_name = 'DefaultPreprocessor'  # -preprocessor_name 参数的默认值
    overwrite_target_spacing = None  # -overwrite_target_spacing 参数的默认值
    overwrite_plans_name = 'nnUNetPlans'  # -overwrite_plans_name 参数的默认值
    configs = ['2d', '3d_fullres', '3d_lowres']  # -c 参数的默认值
    np = [8, 4, 8]  # -np 参数的默认值
    no_pp = False  # --no_pp 参数的默认值，这里设置为 False 以便进行预处理
    from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess
    print("Fingerprint extraction...")
    extract_fingerprints(
        dataset_ids=dataset_ids,
        fingerprint_extractor_class_name=fpe,
        num_processes=npfp,
        check_dataset_integrity=verify_dataset_integrity,
        clean=clean,
        verbose=verbose
    )

    print('Experiment planning...')
    plan_experiments(
        dataset_ids,
        pl,
        gpu_memory_target,
        preprocessor_name,
        overwrite_target_spacing,
        overwrite_plans_name
    )

    if not no_pp:
        print('Preprocessing...')
        preprocess(dataset_ids, overwrite_plans_name, configs, np, verbose)


def train():
    from nnunetv2.run.run_training import run_training
    dataset_name_or_id = '1'
    configuration = '2d'
    fold = '4'
    tr = 'nnUNetTrainer'  # 这是默认值
    p = 'nnUNetPlans'  # 这是默认值
    pretrained_weights = None  # 这是默认值
    num_gpus = 1  # 这是默认值
    use_compressed = False  # 这是默认值
    npz = False  # 这是默认值
    c = False  # 这是默认值
    val = False  # 这是默认值
    disable_checkpointing = False  # 这是默认值
    device_arg = 'mps'  # 从命令行参数中提取
    # 根据提供的-device参数设置设备
    if device_arg == 'cuda':
        # 设置为使用 GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    elif device_arg == 'mps':
        # 设置为使用 CPU
        device = torch.device('cpu')
    else:
        # 设置为使用 Apple M1/M2
        device = torch.device('mps')

    # 调用 run_training 函数
    run_training(
        dataset_name_or_id,
        configuration,
        fold,
        tr,
        p,
        pretrained_weights,
        num_gpus,
        use_compressed,
        npz,
        c,
        val,
        disable_checkpointing,
        device=device
    )


if __name__ == '__main__':
    setup_env()
    # pre_process_database()
    train()
    # print(torch.__version__)
    # print(torch.cuda.is_available())
