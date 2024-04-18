import os
import argparse
import numpy as np
from glob import glob


def gather_position(path_list: list):  # positionをまとめる<pos>-<channel>
    pos_list = []
    for p in path_list:
        filename = os.path.basename(p)
        pos = filename[:filename.find('-')]
        if not pos in pos_list:
            pos_list.append(pos)
    return pos_list


def gather_well_id(filepath_list: list):  # filepath_listからwell_idをまとめる
    well_id_list = []
    for p in filepath_list:
        pos = p[p.rfind('/')+1:][:6]  # 'r11c16f06p01' to 'r11c16'
        plate_name = p[:10]  # 'BR00117010'
        well_id = f"{plate_name }-{pos}"
        if not well_id in well_id_list:
            well_id_list.append(well_id)
    return well_id_list


def create_filepath_list_from_well_plate_path_list(well_plate_path_list: list):  # well_plate_path_listからfilepath_listを生成
    filepath_list = []
    for well_plate_path in well_plate_path_list:
        well_plate_name = os.path.basename(well_plate_path)
        img_path_list = glob(f"{well_plate_path}/Images/*.tiff")
        pos_list = gather_position(img_path_list)
        filepath_list += [f"{well_plate_name}/Images/{pos}" for pos in pos_list]
    return filepath_list


def create_filepath_list_from_well_id(well_id_list: list, whole_filepath_list: list):  # well_id_listからfilepath_listを生成
    filepath_list = []
    for well_id in well_id_list:
        plate_name = well_id[:well_id.find('-')]
        well_name = well_id[well_id.find('-')+1:]

        fil = filter(lambda x: plate_name in x and well_name in x, whole_filepath_list)
        filtered_filepath_list = []
        for f in fil:
            filtered_filepath_list.append(f)

        filepath_list += filtered_filepath_list
    return filepath_list


def write_list_to_txt(savefilepath: str, path_list: list):  # listをtextに書き出し
    with open(savefilepath, 'w') as f:
        for ind, d in enumerate(path_list):
            if ind < len(path_list) - 1:
                f.write("%s\n" % d)
            else:
                f.write("%s" % d)


def write_train_val_list_to_txt(save_dir, fold_num, train_val_path_list):  # train_val_listをfoldに分割してtextに書き出し

    for fd in range(fold_num):

        dist = os.path.join(save_dir, 'fold{}'.format(fd + 1))
        os.makedirs(dist, exist_ok=True)

        train_filepath_list = []
        for pa in train_val_path_list[:int(len(train_val_path_list)/fold_num)*fd]+train_val_path_list[int(len(train_val_path_list)/fold_num)*(fd+1):]:
            train_filepath_list.append(pa)

        val_filepath_list = []
        for pa in train_val_path_list[int(len(train_val_path_list)/fold_num)*fd:int(len(train_val_path_list)/fold_num)*(fd+1)]:
            val_filepath_list.append(pa)

        print(f'[fold{fd}] train: {len(train_filepath_list)}, validation: {len(val_filepath_list)}')
        write_list_to_txt(savefilepath=f"{dist}/train.txt", path_list=train_filepath_list)
        write_list_to_txt(savefilepath=f"{dist}/validation.txt", path_list=val_filepath_list)


def check_not_has_duplicates(seq: list):
    return len(seq) == len(set(seq))


def remove_duplicates(l:list):
    return list(set(l))


def verify_datasets(save_dir, fold_num, total_train_val, total):  # data leakageがないか検証
    # trainとvalidationの被りなしを確認
    print("[Verify that train and validation are not covered]")
    for fd in range(fold_num):
        target = os.path.join(save_dir, 'fold{}'.format(fd + 1))
        with open(os.path.join(target, 'train.txt'), 'r') as f:
            filelist_train = [line.rstrip() for line in f]
        with open(os.path.join(target, 'validation.txt'), 'r') as f:
            filelist_validatioin = [line.rstrip() for line in f]

        matched_list = list(set(filelist_train) & set(filelist_validatioin))
        match_check = True if len(matched_list) == 0 else False
        len_check = len(np.unique(filelist_train + filelist_validatioin)) == total_train_val
        check_duplicates_train = check_not_has_duplicates(filelist_train)
        check_duplicates_val = check_not_has_duplicates(filelist_validatioin)
        print("[fold{}] {}, {}, {}, {}".format(fd, len_check, match_check, check_duplicates_train, check_duplicates_val))

    print("=" * 100)
    print("[Verify that train/validation and test are not covered]")
    # train/validationとtestの被りなしを確認
    with open(os.path.join(save_dir, 'test.txt'), 'r') as f:
        filelist_test = [line.rstrip() for line in f]
    for fd in range(fold_num):
        target = os.path.join(save_dir, 'fold{}'.format(fd + 1))
        with open(os.path.join(target, 'train.txt'), 'r') as f:
            filelist_train = [line.rstrip() for line in f]
        with open(os.path.join(target, 'validation.txt'), 'r') as f:
            filelist_validatioin = [line.rstrip() for line in f]

        matched_list = list(set(filelist_train + filelist_validatioin) & set(filelist_test))
        match_check = True if len(matched_list) == 0 else False
        len_check = len(np.unique(filelist_train + filelist_validatioin + filelist_test)) == total
        check_duplicates_train = check_not_has_duplicates(filelist_train)
        check_duplicates_val = check_not_has_duplicates(filelist_validatioin)
        check_duplicates_test = check_not_has_duplicates(filelist_test)
        print("[fold{}] {}, {}, {}, {}, {}".format(fd, match_check, len_check, check_duplicates_train, check_duplicates_val, check_duplicates_test))


def main(args):
    root_path = str(args.root_path)
    save_dir = str(args.save_path)
    test_ratio = float(args.test_ratio)
    test_split_mode = str(args.test_split_mode)
    fold_num = int(args.fold_num)
    seed = int(args.seed)
    select_wellplate = eval(args.select_wellplate)

    # create save path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set seed
    rg = np.random.default_rng(seed)

    well_plate_path_list = glob(f"{root_path}/images/*")
    if len(select_wellplate) > 0:
        well_plate_path_list = [p for p in well_plate_path_list if os.path.basename(p) in select_wellplate]
        assert len(well_plate_path_list) == len(select_wellplate), \
            f'Cannot select wellplate correctly, {select_wellplate}'

    with open(f'{save_dir}/split_logs.txt', 'w') as f:
        print(f"[Split mode] {test_split_mode}")
        print('='*100)
        if test_split_mode == 'wellplate':
            # select well plate path
            test_well_plate_num = int(np.floor(test_ratio*len(well_plate_path_list)))
            if test_well_plate_num == 0:
                test_well_plate_num = 1
            test_well_plate_path_list = rg.choice(well_plate_path_list, test_well_plate_num).tolist()

            train_val_well_plate_path_list = []
            for well_plate_path in well_plate_path_list:
                if not well_plate_path in test_well_plate_path_list:
                    train_val_well_plate_path_list.append(well_plate_path)
            print(f"total well num (train+validation): {len(train_val_well_plate_path_list)}")
            print(f"total well num (test): {len(test_well_plate_path_list)}")

            # create filepath_list
            train_val_filepath_list = create_filepath_list_from_well_plate_path_list(train_val_well_plate_path_list)
            test_filepath_list = create_filepath_list_from_well_plate_path_list(test_well_plate_path_list)

            # check and remove duplicates
            if not check_not_has_duplicates(train_val_filepath_list):
                train_val_filepath_list = remove_duplicates(train_val_filepath_list)
            if not check_not_has_duplicates(test_filepath_list):
                test_filepath_list = remove_duplicates(test_filepath_list)

            print('total file num: {}'.format(len(train_val_filepath_list) + len(test_filepath_list)))
            print('total file num (train+validation): {}'.format(len(train_val_filepath_list)))
            print('total file num (test): {}'.format(len(test_filepath_list)))

        elif test_split_mode == 'well':
            # create filepath_list
            filepath_list = create_filepath_list_from_well_plate_path_list(well_plate_path_list)

            # gather well id
            well_id_list = gather_well_id(filepath_list)

            # split well id
            test_well_id_num = int(np.floor(test_ratio*len(well_id_list)))
            if test_well_id_num == 0:
                test_well_id_num = 1
            test_well_id_list = rg.choice(well_id_list, test_well_id_num).tolist()

            train_val_well_id_list = []
            for well_id in well_id_list:
                if not well_id in test_well_id_list:
                    train_val_well_id_list.append(well_id)

            # extract filepath
            train_val_filepath_list = create_filepath_list_from_well_id(train_val_well_id_list, filepath_list)
            test_filepath_list = create_filepath_list_from_well_id(test_well_id_list, filepath_list)

            # check and remove duplicates
            if not check_not_has_duplicates(train_val_filepath_list):
                train_val_filepath_list = remove_duplicates(train_val_filepath_list)
            if not check_not_has_duplicates(test_filepath_list):
                test_filepath_list = remove_duplicates(test_filepath_list)

            print('total file num: {}'.format(len(train_val_filepath_list)+len(test_filepath_list)))
            print('total file num (train+validation): {}'.format(len(train_val_filepath_list)))
            print('total file num (test): {}'.format(len(test_filepath_list)))

        elif test_split_mode == 'random':
            # gather img_path
            filepath_list = []
            for well_plate_path in well_plate_path_list:
                filepaths = glob(f"{well_plate_path}/Images/*.tiff")
                filepath_list.append(filepaths)

            # split img_path
            test_filepath_num = int(np.floor(test_ratio*len(filepath_list)))
            if test_filepath_num == 0:
                test_filepath_num = 1
            test_filepath_list = rg.choice(filepath_list, test_filepath_num).tolist()

            train_val_filepath_list = []
            for filepath in filepath_list:
                if not filepath in test_filepath_list:
                    train_val_filepath_list.append(filepath)

            # check and remove duplicates
            if not check_not_has_duplicates(train_val_filepath_list):
                train_val_filepath_list = remove_duplicates(train_val_filepath_list)
            if not check_not_has_duplicates(test_filepath_list):
                test_filepath_list = remove_duplicates(test_filepath_list)

            print('total file num: {}'.format(len(train_val_filepath_list) + len(test_filepath_list)))
            print('total file num (train+validation): {}'.format(len(train_val_filepath_list)))
            print('total file num(test): {}'.format(len(test_filepath_list)))

        else:
            raise NotImplementedError

        # split train/val and save filepath_list
        rg.shuffle(train_val_filepath_list)  # shuffle train val filepath list
        write_train_val_list_to_txt(save_dir=save_dir, fold_num=fold_num, train_val_path_list=train_val_filepath_list)
        write_list_to_txt(savefilepath=f"{save_dir}/test.txt", path_list=test_filepath_list)

        # verify filepath_list
        verify_datasets(save_dir=save_dir, fold_num=fold_num,
                        total_train_val=len(train_val_filepath_list),
                        total=len(train_val_filepath_list+test_filepath_list))

        print('='*100)
        print('Finish')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, help='set data root path')
    parser.add_argument('--save_path', type=str, help='set built dataset save path')
    parser.add_argument('--test_ratio', type=float, help='set test dataset ratio')
    parser.add_argument('--test_split_mode', type=str, choices=['wellplate', 'well', 'random'],
                        help='choice test split mode [wellplate, well random]')
    parser.add_argument('--fold_num', type=int, help='set number of cross-validation fold')
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    parser.add_argument('--select_wellplate', type=str, help='set wellplate names to select', default='[]')
    args = parser.parse_args()

    main(args=args)