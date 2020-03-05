from pipeline.train import train_fastfcn_mod

import argparse

import os.path

import sys


if __name__=='__main__':

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command')

    train_parser = subparsers.add_parser('train', help=train_fastfcn_mod.__doc__)
    train_parser.add_argument(
        '-name', default=None, type=str, required=False,
        help='Nickname for model.')
    train_parser.add_argument(
        '-epochs', default=2, type=int, required=False,
        help='Number of epochs.')
    train_parser.add_argument(
        '-report', default=5, type=int, required=False,
        help='Number of batches between loss reports (int).')
    train_parser.add_argument(
        '-batch_size', default=16, type=int, required=False,
        help='The filter used to match logs.')
    train_parser.add_argument(
        '-train_path', default=None, type=str, required=False,
        help='Folder containing training images, with images and masks subdirectory.')
    train_parser.add_argument(
        '-batch_trim', default=None, type=int, required=False,
        help='Option to only train for a limit number of batches in each epoch.')

    args = parser.parse_args()
    print('Args:\n', args)

    if args.command == 'train':
        train_fastfcn_mod(
            num_epochs=args.epochs, reporting_int=args.report,
            batch_size=args.batch_size, model_nickname=args.name,
            train_path=args.train_path, batch_trim=args.batch_trim
            )p