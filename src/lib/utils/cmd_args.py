import argparse
import configparser

def create_dataset_parser(remaining_argv, **conf_dict):
    # Dataset options
    parser = argparse.ArgumentParser(description='Dataset Parameters', add_help=False)
    parser.set_defaults(**conf_dict)
    parser.add_argument('--root_path',
                        type=str,
                        help='/path/to/dataset', default=conf_dict['root_path'])
    args, remaining_argv = parser.parse_known_args(remaining_argv)

    return parser, args, remaining_argv


def create_model_parser(remaining_argv, **conf_dict):
    # Model options
    parser = argparse.ArgumentParser(description='Model Parameters', add_help=False)
    parser.set_defaults(**conf_dict)

    parser.add_argument('--model',
                        help='model name', default=conf_dict['model'])
    args, remaining_argv = parser.parse_known_args(remaining_argv)

    return parser, args, remaining_argv


def create_runtime_parser(remaining_argv, **conf_dict):
    # Runtime options
    parser = argparse.ArgumentParser(description='Runtime Parameters', add_help=False)
    parser.set_defaults(**conf_dict)
    parser.add_argument('--save_dir',
                        help='Save directory', default=conf_dict['save_dir'])
    parser.add_argument('--model_dir',
                        help='Model directory which trained files are saved', default=conf_dict['model_dir'])
    parser.add_argument('--device',
                        help='CPU/GPU ID', default=conf_dict['device'])
    args, remaining_argv = parser.parse_known_args(remaining_argv)

    return parser, args, remaining_argv


def config_paraser():
    conf_parser = argparse.ArgumentParser(
        description=__doc__,  # printed with -h/--help
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False
    )
    conf_parser.add_argument("-c", "--conf_file", help="Specify config file", metavar="FILE_PATH")
    args, remaining_argv = conf_parser.parse_known_args()

    dataset_conf_dict = {}
    model_conf_dict = {}
    runtime_conf_dict = {}

    if args.conf_file is not None:
        config = configparser.ConfigParser()
        config.read([args.conf_file])
        dataset_conf_dict = dict(config.items("Dataset"))
        model_conf_dict = dict(config.items("Model"))
        runtime_conf_dict = dict(config.items("Runtime"))

    ''' Parameters '''
    # Dataset options
    dataset_parser, dataset_args, remaining_argv = create_dataset_parser(remaining_argv, **dataset_conf_dict)
    # Model options
    model_parser, model_args, remaining_argv = create_model_parser(remaining_argv, **model_conf_dict)
    # Runtime options
    runtime_parser, runtime_args, remaining_argv = create_runtime_parser(remaining_argv, **runtime_conf_dict)

    # merge options
    parser = argparse.ArgumentParser(
        parents=[conf_parser, dataset_parser, model_parser, runtime_parser])
    args = parser.parse_args()

    return args
