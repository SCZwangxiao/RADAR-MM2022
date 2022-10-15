import os
import random
import multiprocessing
import time
import sys
import argparse


def create_proc(args):
    config, gpu_locks = args
    idx = -1
    LOCK = None
    while idx < 0:
        for i, lock in enumerate(gpu_locks):
            if lock.acquire(False):
                idx = i
                LOCK = lock
                break
    gpu, addr, port = settings[idx]
    ######################################
    print(f"{time.asctime(time.localtime(time.time()))} Training {config} on GPU-{gpu} BEGINS...")
    sys.stdout.flush()
    
    os.system(f"python tools/train.py {config} --launcher pytorch --gpu-ids {gpu} --validate --test-best")
    LOCK.release()
    print(f"{time.asctime(time.localtime(time.time()))} Training {config} on GPU-{gpu} DONE!")
    sys.stdout.flush()

def parse_args():
    parser = argparse.ArgumentParser(
        description='Grid Search')
    parser.add_argument(
        '--num-gpu',
        default=4,
        type=int,
        help='number of gpus')
    parser.add_argument(
        '--shuffle',
        action='store_true',
        help='whether to shuffle to avoid skewed workload')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    num_gpu = args.num_gpu
    shuffle = args.shuffle
    ##### Change Settings #######
    SERVER = 'kwai'
    settings = [(f"{i}", f"127.0.0.{i+1}", f"{29501+i}") for i in range(num_gpu)]
    #USER = 'root'

    #### Configs and Parameters ####
    configs=[
        # 'configs/recommendation/ourgnn/ourgnn_l2_kwai_cosmetic.py --cfg-options model.linker.lambda0=0.0 --work-dir ./work_dirs/ourgnn_l2_kwai_cosmetic_lambda0/',
        # 'configs/recommendation/ourgnn/ourgnn_l2_kwai_cosmetic.py --cfg-options model.linker.lambda0=0.0001 --work-dir ./work_dirs/ourgnn_l2_kwai_cosmetic_lambda0.0001/',
        # 'configs/recommendation/ourgnn/ourgnn_l2_kwai_cosmetic.py --cfg-options model.linker.lambda0=0.0005 --work-dir ./work_dirs/ourgnn_l2_kwai_cosmetic_lambda0.0005/',
        # 'configs/recommendation/ourgnn/ourgnn_l2_kwai_cosmetic.py --cfg-options model.linker.lambda0=0.001 --work-dir ./work_dirs/ourgnn_l2_kwai_cosmetic_lambda0.001/',
        # 'configs/recommendation/ourgnn/ourgnn_l2_kwai_cosmetic.py --cfg-options model.linker.lambda0=0.005 --work-dir ./work_dirs/ourgnn_l2_kwai_cosmetic_lambda0.005/',
        # 'configs/recommendation/ourgnn/ourgnn_l2_kwai_cosmetic.py --cfg-options model.linker.lambda0=0.01 --work-dir ./work_dirs/ourgnn_l2_kwai_cosmetic_lambda0.01/',
        # 'configs/recommendation/ourgnn/ourgnn_l2_kwai_cosmetic.py --cfg-options model.linker.lambda0=0.025 --work-dir ./work_dirs/ourgnn_l2_kwai_cosmetic_lambda0.025/',
        # 'configs/recommendation/ourgnn/ourgnn_l2_kwai_cosmetic.py --cfg-options model.linker.lambda0=0.05 --work-dir ./work_dirs/ourgnn_l2_kwai_cosmetic_lambda0.05/',
        # 'configs/recommendation/ourgnn/ourgnn_l2_kwai_cosmetic.py --cfg-options model.linker.lambda0=0.1 --work-dir ./work_dirs/ourgnn_l2_kwai_cosmetic_lambda0.1/',
        # 'configs/recommendation/ourgnn/ourgnn_l2_kwai_cosmetic.py --cfg-options model.linker.lambda0=0.5 --work-dir ./work_dirs/ourgnn_l2_kwai_cosmetic_lambda0.5/',
        # 'configs/recommendation/ourgnn/ourgnn_l2_kwai_cosmetic.py --cfg-options model.linker.lambda0=1 --work-dir ./work_dirs/ourgnn_l2_kwai_cosmetic_lambda1/',
        # 'configs/recommendation/ourgnn/ourgnn_l2_kwai_apparel.py --cfg-options model.linker.lambda0=0.0 --work-dir ./work_dirs/ourgnn_l2_kwai_apparel_lambda0/',
        # 'configs/recommendation/ourgnn/ourgnn_l2_kwai_apparel.py --cfg-options model.linker.lambda0=0.0001 --work-dir ./work_dirs/ourgnn_l2_kwai_apparel_lambda0.0001/',
        # 'configs/recommendation/ourgnn/ourgnn_l2_kwai_apparel.py --cfg-options model.linker.lambda0=0.0005 --work-dir ./work_dirs/ourgnn_l2_kwai_apparel_lambda0.0005/',
        # 'configs/recommendation/ourgnn/ourgnn_l2_kwai_apparel.py --cfg-options model.linker.lambda0=0.001 --work-dir ./work_dirs/ourgnn_l2_kwai_apparel_lambda0.001/',
        # 'configs/recommendation/ourgnn/ourgnn_l2_kwai_apparel.py --cfg-options model.linker.lambda0=0.005 --work-dir ./work_dirs/ourgnn_l2_kwai_apparel_lambda0.005/',
        # 'configs/recommendation/ourgnn/ourgnn_l2_kwai_apparel.py --cfg-options model.linker.lambda0=0.01 --work-dir ./work_dirs/ourgnn_l2_kwai_apparel_lambda0.01/',
        # 'configs/recommendation/ourgnn/ourgnn_l2_kwai_apparel.py --cfg-options model.linker.lambda0=0.025 --work-dir ./work_dirs/ourgnn_l2_kwai_apparel_lambda0.025/',
        # 'configs/recommendation/ourgnn/ourgnn_l2_kwai_apparel.py --cfg-options model.linker.lambda0=0.05 --work-dir ./work_dirs/ourgnn_l2_kwai_apparel_lambda0.05/',
        # 'configs/recommendation/ourgnn/ourgnn_l2_kwai_apparel.py --cfg-options model.linker.lambda0=0.1 --work-dir ./work_dirs/ourgnn_l2_kwai_apparel_lambda0.1/',
        # 'configs/recommendation/ourgnn/ourgnn_l2_kwai_apparel.py --cfg-options model.linker.lambda0=0.5 --work-dir ./work_dirs/ourgnn_l2_kwai_apparel_lambda0.5/',
        # 'configs/recommendation/ourgnn/ourgnn_l2_kwai_apparel.py --cfg-options model.linker.lambda0=1 --work-dir ./work_dirs/ourgnn_l2_kwai_apparel_lambda1/',
        'configs/recommendation/ourgnn/ourgnn_l2_kwai_food.py --cfg-options model.linker.lambda0=0.0 --work-dir ./work_dirs/ourgnn_l2_kwai_food_lambda0/',
        'configs/recommendation/ourgnn/ourgnn_l2_kwai_food.py --cfg-options model.linker.lambda0=0.0001 --work-dir ./work_dirs/ourgnn_l2_kwai_food_lambda0.0001/',
        'configs/recommendation/ourgnn/ourgnn_l2_kwai_food.py --cfg-options model.linker.lambda0=0.0005 --work-dir ./work_dirs/ourgnn_l2_kwai_food_lambda0.0005/',
        'configs/recommendation/ourgnn/ourgnn_l2_kwai_food.py --cfg-options model.linker.lambda0=0.001 --work-dir ./work_dirs/ourgnn_l2_kwai_food_lambda0.001/',
        'configs/recommendation/ourgnn/ourgnn_l2_kwai_food.py --cfg-options model.linker.lambda0=0.005 --work-dir ./work_dirs/ourgnn_l2_kwai_food_lambda0.005/',
        'configs/recommendation/ourgnn/ourgnn_l2_kwai_food.py --cfg-options model.linker.lambda0=0.01 --work-dir ./work_dirs/ourgnn_l2_kwai_food_lambda0.01/',
        'configs/recommendation/ourgnn/ourgnn_l2_kwai_food.py --cfg-options model.linker.lambda0=0.025 --work-dir ./work_dirs/ourgnn_l2_kwai_food_lambda0.025/',
        'configs/recommendation/ourgnn/ourgnn_l2_kwai_food.py --cfg-options model.linker.lambda0=0.05 --work-dir ./work_dirs/ourgnn_l2_kwai_food_lambda0.05/',
        'configs/recommendation/ourgnn/ourgnn_l2_kwai_food.py --cfg-options model.linker.lambda0=0.1 --work-dir ./work_dirs/ourgnn_l2_kwai_food_lambda0.1/',
        'configs/recommendation/ourgnn/ourgnn_l2_kwai_food.py --cfg-options model.linker.lambda0=0.5 --work-dir ./work_dirs/ourgnn_l2_kwai_food_lambda0.5/',
        'configs/recommendation/ourgnn/ourgnn_l2_kwai_food.py --cfg-options model.linker.lambda0=1 --work-dir ./work_dirs/ourgnn_l2_kwai_food_lambda1/',
        'configs/recommendation/ourgnn/ourgnn_l2_kwai_sports.py --cfg-options model.linker.lambda0=0.0 --work-dir ./work_dirs/ourgnn_l2_kwai_sports_lambda0/',
        'configs/recommendation/ourgnn/ourgnn_l2_kwai_sports.py --cfg-options model.linker.lambda0=0.0001 --work-dir ./work_dirs/ourgnn_l2_kwai_sports_lambda0.0001/',
        'configs/recommendation/ourgnn/ourgnn_l2_kwai_sports.py --cfg-options model.linker.lambda0=0.0005 --work-dir ./work_dirs/ourgnn_l2_kwai_sports_lambda0.0005/',
        'configs/recommendation/ourgnn/ourgnn_l2_kwai_sports.py --cfg-options model.linker.lambda0=0.001 --work-dir ./work_dirs/ourgnn_l2_kwai_sports_lambda0.001/',
        'configs/recommendation/ourgnn/ourgnn_l2_kwai_sports.py --cfg-options model.linker.lambda0=0.005 --work-dir ./work_dirs/ourgnn_l2_kwai_sports_lambda0.005/',
        'configs/recommendation/ourgnn/ourgnn_l2_kwai_sports.py --cfg-options model.linker.lambda0=0.01 --work-dir ./work_dirs/ourgnn_l2_kwai_sports_lambda0.01/',
        'configs/recommendation/ourgnn/ourgnn_l2_kwai_sports.py --cfg-options model.linker.lambda0=0.025 --work-dir ./work_dirs/ourgnn_l2_kwai_sports_lambda0.025/',
        'configs/recommendation/ourgnn/ourgnn_l2_kwai_sports.py --cfg-options model.linker.lambda0=0.05 --work-dir ./work_dirs/ourgnn_l2_kwai_sports_lambda0.05/',
        'configs/recommendation/ourgnn/ourgnn_l2_kwai_sports.py --cfg-options model.linker.lambda0=0.1 --work-dir ./work_dirs/ourgnn_l2_kwai_sports_lambda0.1/',
        'configs/recommendation/ourgnn/ourgnn_l2_kwai_sports.py --cfg-options model.linker.lambda0=0.5 --work-dir ./work_dirs/ourgnn_l2_kwai_sports_lambda0.5/',
        'configs/recommendation/ourgnn/ourgnn_l2_kwai_sports.py --cfg-options model.linker.lambda0=1 --work-dir ./work_dirs/ourgnn_l2_kwai_sports_lambda1/',
    ]
    #################################
    if shuffle:
        random.shuffle(configs)
    gpu_managers = [multiprocessing.Manager() for _ in range(len(settings))]
    gpu_locks = [manager.Lock() for manager in gpu_managers]

    pool = multiprocessing.Pool(processes=len(settings))
    print('########################################')
    print(len(settings))
    args_list = []

    for i, config in enumerate(configs):
        print('########################################')
        print(i, config)
        args_list.append((config, gpu_locks))

    pool.map(create_proc, args_list, chunksize=1)
    pool.close()
    pool.join()
    print("ALL TASKS DONE")