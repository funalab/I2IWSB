import os
import argparse
import numpy as np

def gather_pos(filepath_list: list):
    pos_dict = {}
    for filepath in filepath_list:
        # e.g.) filepath = 'BR00117011__2020-11-08T19_57_47-Measurement1/Images/r09c16f09p01'
        dirpath = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        pos_tmp = filename[:filename.rfind('f')]
        pos = f"{dirpath}/{pos_tmp}"
        site = filename[filename.rfind('f'):filename.rfind(r'f')+3]
        # create key
        if not pos in list(pos_dict.keys()):
            pos_dict[pos] = [site]
        else:
            if not site in pos_dict[pos]:
                pos_dict[pos].append(site)
    return pos_dict

def load_filepath_list(txt_path):
    filepath_list = []
    with open(txt_path, 'r') as f:
        for line in f:
            filepath_list.append(line.replace('\n', ''))
    filepath_list = [p for p in filepath_list if p != ""]
    return filepath_list


def choice_site(rg, filepath_list: list, pos_dict: dict, num_site_per_well: int):
    candidates = []
    for pos in list(pos_dict.keys()):
        sites = pos_dict[pos]
        choiced_sites = rg.choice(sites, num_site_per_well)
        for choiced_site in choiced_sites:
            candidate = f"{pos}{choiced_site}p01"
            candidates.append(candidate)

    # validate
    choiced_filepath_list = []
    for path in candidates:
        if path in filepath_list: # f"{path}\n"
            choiced_filepath_list.append(path)

    return choiced_filepath_list


def write_list_to_txt(savefilepath: str, path_list: list):
    with open(savefilepath, 'w') as f:
        for ind, d in enumerate(path_list):
            if ind < len(path_list) - 1:
                f.write("%s\n" % d)
            else:
                f.write("%s" % d)


def main(args):
    test_path = str(args.test_path)
    num_site_per_well = int(args.num_site_per_well)
    seed = int(args.seed)

    # set seed
    rg = np.random.default_rng(seed)

    # load test
    test_filepath_list = load_filepath_list(txt_path=test_path)

    # gather position
    pos_dict = gather_pos(filepath_list=test_filepath_list)

    # choice site
    extracted_filepath_list = choice_site(rg, test_filepath_list, pos_dict, num_site_per_well)
    print(f"test: {len(test_filepath_list)}")
    print(f"test (extracted): {len(extracted_filepath_list)}")

    # save
    svaefilepath = test_path.replace('test.txt', f'test_extracted_{num_site_per_well}_site_per_well.txt')
    write_list_to_txt(svaefilepath, extracted_filepath_list)
    print('finish')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, help='set test path')
    parser.add_argument('--num_site_per_well', type=str, help='num_site_per_well')
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    args = parser.parse_args()

    main(args=args)