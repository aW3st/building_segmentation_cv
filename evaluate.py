import torch
from torch.nn.functional import softmax
from PIL import Image
import numpy as np
import os

from torchvision import transforms
import pdb

from FastFCN.encoding.models import MultiEvalModule
import FastFCN.encoding.utils as utils
from pipeline.fastfcn_modified import get_model
from pipeline.train import ObjectView

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


    path_to_state_dict = f'models/{model_name}/{model_name}.pt'
    model.load_state_dict(torch.load(path_to_state_dict))

    return model


def get_single_pred(model, img_name=None, img_path = None):
    '''
    Predict for a single image.
    '''
    
    test_img_dir = 'submission_data/test'

    # Predict on a sample image.
    if img_path is None:
        if img_name is None:
            img_name = '0a0a36'
        img_path = os.path.join(test_img_dir, f'{img_name}/{img_name}.tif')

    img = Image.open(img_path).convert("RGB")


    img_tensor = transforms.functional.to_tensor(img)
    output = model(img_tensor.view(-1, 3, 1024, 1024))[2]
    np_pred = torch.max(output, 1)[1].cpu().numpy() * 255
    out_img = Image.fromarray(np_pred.squeeze().astype('uint8'))
    out_img.show()
    

    def predict_quadrant(full_tensor, quadrant=2):
        
        if quadrant==1:
            top, left = 512, 0
        elif quadrant==2:
            top, left = 0, 0
        elif quadrant==3:
            top, left = 0, 512
        elif quadrant==4:
            top, left = 512, 512

        quad = transforms.functional.crop(full_img, top, left, 512, 512)
        quad = quad.view(-1, 3, 512, 512) 

        output = model(quad)[2]
        np_pred = torch.max(output, 1)[1].cpu().numpy() * 255
        out_img = Image.fromarray(np_pred.squeeze().astype('uint8'))

        # pdb.set_trace()

        return out_img


    merged = Image.new('RGB', (1024, 1024), (0,0,0))
    
    merged.paste(
        predict_quadrant(img_tensor, quadrant=1),
        (512, 0)
    )

    merged.paste(
        predict_quadrant(img_tensor, quadrant=2),
        (0,0)
    )

    merged.paste(
        predict_quadrant(img_tensor, quadrant=3),
        (0, 512)
    )

    merged.paste(
        predict_quadrant(img_tensor, quadrant=4),
        (512,512)
    )

    return merged


def predict_test_set(model, path_to_test_set, output_path='model_outs/'):
    '''
    Predict for the entire submission set.
    '''

    raise NotImplementedError


if __name__ == "__main__":

    MODEL_NAME = 'model_test'
    MODEL = load_model_with_weights(model_name=MODEL_NAME)
    MODEL.eval()

    # Predict a single test image:
    IMG_NAME = '0a0a36'
    PRED = get_single_pred(MODEL, img_name=IMG_NAME)
    PRED.show()

    IMAGE_OUT_PATH = f'models/{MODEL_NAME}/predictions/{IMG_NAME}.tif'
    # Save test image to correct model output directory.
    PRED.save(IMAGE_OUT_PATH)