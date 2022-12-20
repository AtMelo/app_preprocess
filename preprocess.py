import argparse
import logging
import yaml

from utils import setup_logger, Preprocess


def parse_opt():
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('--cfg',
                        type=str,
                        default='cfgs/preproc_cfg.yaml',
                        help='Path to preprocess config')
    parser.add_argument('--prepare',
                        action='store_true',
                        help='Prepare data for the next transformations')
    parser.add_argument('--name_cls',
                        nargs='?',
                        type=str,
                        default='class_name',
                        help='Enter name of the class')
    parser.add_argument('--form_dataset',
                        action='store_true',
                        help='Expand in (num_transform + 1) times ans split '
                             'augment image to train, val and test folder')
    parser.add_argument('--num_transform',
                        type=int,
                        default=None,
                        help='Expand dataset with albumentatons transform')
    parser.add_argument('--upd_cls_txt',
                        action='store_true',
                        help='Update all files of classes.txt')
    parser.add_argument('--view',
                        action='store_true',
                        help='View transform images')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Verbose logging')

    opt = parser.parse_args()

    LVL = logging.DEBUG if opt.verbose else logging.INFO
    setup_logger(name='Preprocess', level=LVL)
    logging.debug(f'Input options: {vars(opt)}')
    return opt


def main(opt):
    # Non-bool parameters
    cfg, name_cls, num_transform = \
        opt.cfg, opt.name_cls, opt.num_transform

    with open(cfg) as f:
        config = yaml.safe_load(f)
    # For preparing data:
    path_initial_img = config.get('path_initial_img')
    path_preproc_img = config.get('path_preproc_img')
    path_dataset = config.get('path_dataset')

    # For creating a transformed dataset
    size_train = config.get('size_train')
    size_val = config.get('size_val')
    size_test = config.get('size_test')
    names = config.get('names')

    preprocessing = Preprocess(path_initial_img,
                               path_preproc_img,
                               path_dataset,
                               name_cls)
    if opt.prepare:
        logging.info('\tRenames processing...')
        preprocessing.prepare_data()
        logging.info('\t...executed successfully')
        logging.info('GREAT JOB! NEXT STEP MARKING IMAGES')
        raise SystemExit('Use LabelImg.py for marking images')

    if opt.form_dataset:
        logging.info(f'\tExpanding in {num_transform+1} times and split '
                     f'dataset (train:{size_train}, val:{size_val}, '
                     f'test:{size_test})...')
        preprocessing.expand_split_dataset(num_transform,
                                           size_train,
                                           size_val,
                                           size_test,
                                           opt.view)
        logging.info('\t...executed successfully')
    if opt.upd_cls_txt:
        logging.info('\tUpdate all files of classes.txt...')
        preprocessing.update_classes_txt(names_cls=names)
        logging.info('\t...executed successfully')


if __name__ == "__main__":
    options = parse_opt()
    main(options)
