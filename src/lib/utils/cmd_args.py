import argparse
import configparser

def create_dataset_parser(remaining_argv, **conf_dict):
    # Dataset options
    parser = argparse.ArgumentParser(description='Dataset Parameters', add_help=False)
    parser.set_defaults(**conf_dict)
    parser.add_argument('--root_path',
                        type=str,
                        help='/path/to/dataset')
    parser.add_argument('--split_list_train',
                        type=str,
                        help='/path/to/train_list.txt (list of {train,validation} files)')
    parser.add_argument('--split_list_validation',
                        type=str,
                        help='/path/to/validation_list.txt (list of {train,validation} files)')
    parser.add_argument('--basename',
                        type=str,
                        help='basename of input directory')
    args, remaining_argv = parser.parse_known_args(remaining_argv)

    return parser, args, remaining_argv


def create_model_parser(remaining_argv, **conf_dict):
    # generator class options
    parser = argparse.ArgumentParser(description='Model Parameters', add_help=False)
    parser.set_defaults(**conf_dict)

    parser.add_argument('--init_classifier',
                        help='Initialize the Classifier file')
    parser.add_argument('--model',
                        help='model name')
    parser.add_argument('--n_input_channels',
                        help='input channel')
    parser.add_argument('--n_classes',
                        help='classes num')
    parser.add_argument('--lossfun',
                        help='loss function name')
    parser.add_argument('--eval_metrics',
                        help='metric')
    parser.add_argument('--eval_maximize',
                        help='boolean to maximize metric')
    args, remaining_argv = parser.parse_known_args(remaining_argv)

    return parser, args, remaining_argv


def create_runtime_parser(remaining_argv, **conf_dict):
    # Model runtime options (for adversarial network model)
    parser = argparse.ArgumentParser(description='Runtime Parameters', add_help=False)
    parser.set_defaults(**conf_dict)
    parser.add_argument('--save_dir',
                        help='Root directory which trained files are saved')
    parser.add_argument('--batchsize',
                        help='Learning minibatch size, default=32')
    parser.add_argument('--val_batchsize',
                        help='Validation minibatch size')
    parser.add_argument('--epoch',
                        help='Number of epochs to train, default=10')
    parser.add_argument('--optimizer',
                        help='Optimizer name for generator')
    parser.add_argument('--lr',
                        help='Initial learning rate ("alpha" in case of Adam)')
    parser.add_argument('--momentum',
                        help='Momentum')
    parser.add_argument('--weight_decay',
                        help='Weight decay for optimizer scheduling')
    parser.add_argument('--device',
                        help='CPU/GPU ID')
    parser.add_argument('--phase',
                        help='Specify mode (train, test)')
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
    # Modeloptions
    model_parser, model_args, remaining_argv = create_model_parser(remaining_argv, **model_conf_dict)
    # Runtime options
    runtime_parser, runtime_args, remaining_argv = create_runtime_parser(remaining_argv, **runtime_conf_dict)

    # merge options
    parser = argparse.ArgumentParser(
        parents=[conf_parser, dataset_parser, model_parser, runtime_parser])
    args = parser.parse_args()

    return args
