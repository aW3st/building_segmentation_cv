import torch
from torch.nn.functional import softmax
from PIL import Image
import numpy as np
import os

from torchvision import transforms
import pdb

import pipeline.fastfcn_modified as fcn_mod

from tqdm import tqdm

from FastFCN.encoding.models import MultiEvalModule
import FastFCN.encoding.utils as utils
from pipeline.fastfcn_modified import get_model
from pipeline.train import ObjectView

import argparse

def img_frombytes(data):
    size = data.shape[::-1]
    databytes = np.packbits(data, axis=1)
    return Image.frombytes(mode='1', size=size, data=databytes)

def load_model_with_weights(model_name=None):
    '''
    Load a model by name from the /models subdirectory.
    '''

    options = {
        'model': 'encnet', # model name (default: encnet)
        'backbone': 'resnet50', # backbone name (default: resnet50)
        'jpu': True, # 'JPU'
        'dilated': False, # 'dilation'
        'lateral': False, #'employ FPN')
        'dataset':'ade20k', # 'dataset name (default: pascal12)')
        'workers': 16, # dataloader threads
        'base_size': 520, # 'base image size'
        'crop_size': 480, # 'crop image size')
        'train_split':'train', # 'dataset train split (default: train)'

        # training hyper params
        'aux': True, # 'Auxilary Loss'
        'aux_weight': 0.2, # 'Auxilary loss weight (default: 0.2)'
        'se_loss': True, # 'Semantic Encoding Loss SE-loss'
        'se_weight': 0.2, # 'SE-loss weight (default: 0.2)'
        'epochs': None, # 'number of epochs to train (default: auto)'
        'start_epoch': 0, # 'start epochs (default:0)'
        'batch_size': 4, # 'input batch size for training (default: auto)'
        'test_batch_size': None, # 'input batch size for testing (default: same as batch size)'

        # optimizer params
        'lr': None, # 'learning rate (default: auto)'
        'lr_scheduler': 'poly', # 'learning rate scheduler (default: poly)'
        'momentum': 0.9, # 'momentum (default: 0.9)'
        'weight_decay': 1e-4, # 'w-decay (default: 1e-4)'

        # cuda, seed and logging
        'no_cuda': False, # 'disables CUDA training'
        'seed': 1, # 'random seed (default: 1)'

        # checking point
        'resume': None, # 'put the path to resuming file if needed'
        'checkname': 'default', # 'set the checkpoint name'
        'model-zoo': None, # 'evaluating on model zoo model'

        # finetuning pre-trained models
        'ft': False, # 'finetuning on a different dataset'

        # evaluation option
        'split': 'val',
        'mode': 'testval',
        'ms': False, # 'multi scale & flip'
        'no_val': False, # 'skip validation during training'
        'save-folder': 'experiments/segmentation/results', # 'path to save images'
    }

    args = ObjectView(options)
    model = get_model(args)

    if model_name[-5:] == '_m.pt':
        model_name = model_name[:-5]

    path_to_state_dict = f'models/{model_name}/{model_name}_m.pt'
    model.load_state_dict(torch.load(path_to_state_dict))

    return model


def output_to_pred_imgs(output, dim=0):

    np_pred = torch.max(output, dim=dim)[1].cpu().numpy()
    return img_frombytes(np_pred)


def get_single_pred(model, img_name=None, img_path = None):
    '''
    Predict for a single image.
    '''
    
    test_img_dir = 'submission_data/test'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Predict on a sample image.
    if img_path is None:
        if img_name is None:
            img_name = '0a0a36'
        img_path = os.path.join(test_img_dir, f'{img_name}/{img_name}.tif')

    img = Image.open(img_path)
    img_tensor = transforms.functional.to_tensor(img)[:3].to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor.view(1, 3, 1024, 1024))[2]
 
    out_img = output_to_pred_imgs(output)

    return out_img


def predict_test_set(model, model_name):
    '''
    Predict for the entire submission set.
    '''

    test_data = fcn_mod.get_dataloader(load_test=True, batch_size=8)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    if not os.path.exists(f'models/{model_name}/predictions'):
            os.makedirs(f'models/{model_name}/predictions')

    print('Beginning Prediction Loop')
    for i, (image_tensors, img_names) in tqdm(enumerate(test_data)):
        
        
        batch_done = True
        for i, name in enumerate(img_names):
            if ~os.path.exists(f'models/{model_name}/predictions/{name}.tif'):
                batch_done = False
        
        if batch_done == True:
            continue
                

        # Load tensors to GPU
        image_tensors = image_tensors.to(device)

        # Predict on image tensors
        with torch.no_grad():
            outputs = model(image_tensors)[2]
            predict_imgs = [output_to_pred_imgs(output) for output in outputs]

        # Zip images, and save.
        for predict_img, img_name in zip(predict_imgs, img_names):
            image_out_path = f'models/{model_name}/predictions/{img_name}.tif'
            predict_img.save(image_out_path)

    return None

if __name__=='__main__':
    try:
        print("Clearing cuda cache.")
        torch.cuda.empty_cache()
        print("Cude cache emptied.")
    except:
        print("No cuda cache to empty.")

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command')

    test_parser = subparsers.add_parser('test', help=predict_test_set.__doc__)
    test_parser.add_argument(
        '-model_name', default=None, type=str, required=True,
        help='Name of model weights file in models directory')
    test_parser.add_argument(
        '-img_dir', default='submissiondata/test', type=str, required=False,
        help='Directory of images to predict.')
    test_parser.add_argument(
        '-overwrite', default=False, type=bool, required=False,
        help='If True, write over existing images. \
                Default behavior checks whether images exist in prediction directory.')

    args = parser.parse_args()

    if args.command == 'test':
        MODEL = load_model_with_weights(model_name=args.model_name)
        MODEL.eval()
        predict_test_set(model=MODEL, model_name=args.model_name)

    else:
        MODEL_NAME = '04-03-2020__Wed__03-04__five_epoch_single_region'
        MODEL = load_model_with_weights(model_name=MODEL_NAME)
        MODEL.eval()
        predict_test_set(model=MODEL, model_name=MODEL_NAME)

    
