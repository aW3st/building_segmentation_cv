import csv
import torch
from PIL import Image
import numpy as np
import sys,os

from torchvision import transforms

from tqdm import tqdm
import pdb

import pipeline.criterion as Criterion
from pipeline.load import get_dataloader
import pipeline.network as Network
import LovaszSoftmax.pytorch.lovasz_losses as L

import argparse

CITY_REGIONS = {
    'znz': {
        '076995': 1701,
        '75cdfa': 2304,
        '425403': 1444,
        '33cae6': 1106,
        '06f252': 2387,
        'e52478': 601,
        'c7415c': 1346,
        'bc32f1': 1521,
        '3f8360': 1086,
        'aee7fd': 1521,
        '9b8638': 1482,
        'bd5c14': 1369,
        '3b20d4': 1849},
    'dar': {
        '353093': 1364,
        'f883a0': 4094,
        '0a4c40': 1604,
        '42f235': 2257,
        'a017f9': 1642,
        'b15fce': 1395},
    'acc': {
        '665946': 6982,
        'a42435': 1318,
        'ca041a': 1566,
        'd41d81': 737},
    'ptn': {'abe1a3': 31, 'f49f31': 13},
    'kam': {'4e7c7f': 868},
    'mon': {
        '401175': 192,
        '493701': 280,
        'f15272': 205,
        '207cc7': 126},
    'nia': {'825a50': 65}
    }

class ObjectView:
    '''
    Helper class to access dict values as attributes.

    Replaces command-line arg-parse options.
    '''
    def __init__(self, d):
        self.__dict__ = d


def img_frombytes(data):
    size = data.shape[::-1]
    databytes = np.packbits(data, axis=1)
    return Image.frombytes(mode='1', size=size, data=databytes)

def load_model_with_weights(model_name=None, num_epochs=8, batch_size=16, use_lovasz=True, se_loss=False, aux=False):
    '''
    Load a model by name from the /models subdirectory.
    '''

    options = {
        'use_jaccard': True,
        'use_lovasz': use_lovasz,
        'early_stopping': False,
        'validation': True,
        'model': 'encnet', # model name (default: encnet)
        'backbone': 'resnet50', # backbone name (default: resnet50)
        'jpu': True, # 'JPU'
        'dilated': True, # 'dilation'
        'lateral': False, #'employ FPN')
        'dataset':'ade20k', # 'dataset name (default: pascal12)')
        'workers': 16, # dataloader threads
        'base_size': 520, # 'base image size'
        'crop_size': 480, # 'crop image size')
        'train_split':'train', # 'dataset train split (default: train)'

        # training hyper params
        'aux': aux, # 'Auxilary Loss'
        'aux_weight': 0.2, # 'Auxilary loss weight (default: 0.2)'
        'se_loss': se_loss, # 'Semantic Encoding Loss SE-loss'
        'se_weight': 0.2, # 'SE-loss weight (default: 0.2)'
        'start_epoch': 0, # 'start epochs (default:0)'
        'batch_size': batch_size, # 'input batch size for training (default: auto)'
        'test_batch_size': None, # 'input batch size for testing (default: same as batch size)'
        'epochs':num_epochs,

        # optimizer params
        'optimizer': 'sgd',
        'lovasz_hinge': True,
        'lr': None, # 'learning rate (default: auto)'
        'lr_scheduler': 'poly', # 'learning rate scheduler (default: poly)'
        'momentum': 0.9, # 'momentum (default: 0.9)'
        'weight_decay': 1e-4, # 'w-decay (default: 1e-4)'

        # cuda, seed and logging
        'no_cuda': False, # 'disables CUDA training'
        'seed': 100, # 'random seed (default: 1)'

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
    }

    model_args = ObjectView(options)
    model = Network.get_model(model_args)

    if model_name[-5:] == '_m.pt':
        model_name = model_name[:-5]

    path_to_state_dict = 'models/{}/{}_m.pt'.format(model_name, model_name)
    model.load_state_dict(torch.load(path_to_state_dict))

    return model


def output_to_pred_imgs(output, dim=0, use_lovasz=False):
    
    if use_lovasz:
        np_pred = (output[0]>0).squeeze().cpu()
    else:
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
        img_path = os.path.join(test_img_dir, '{}/{}.tif'.format(img_name, img_name))

    img = Image.open(img_path)
    img_tensor = transforms.functional.to_tensor(img)[:3].to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor.view(1, 3, 1024, 1024))[2]
 
    out_img = output_to_pred_imgs(output)

    return out_img


def predict_test_set(model, model_name, overwrite=False, use_lovasz=False):
    '''
    Predict for the entire submission set.
    '''

    out_dir = 'models/{}/predictions'.format(model_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    test_dataloader = get_dataloader(load_test=True, batch_size=4, overwrite=overwrite, out_dir=out_dir)
    if test_dataloader == False:
        print('Exiting...')
        sys.exit()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    print('Beginning Prediction Loop')
    tbar = tqdm(test_dataloader)
    for _, (image_tensors, img_names) in enumerate(tbar):        

        # Load tensors to GPU
        image_tensors = image_tensors.to(device)

        # Predict on image tensors
        with torch.no_grad():
            outputs = model(image_tensors)[0]
            predict_imgs = [output_to_pred_imgs(output,use_lovasz=use_lovasz) for output in outputs]

        # Zip images, and save.
        for predict_img, img_name in zip(predict_imgs, img_names):
            image_out_path = 'models/{}/predictions/{}.tif'.format(model_name, img_name)
            predict_img.save(image_out_path, compression="tiff_deflate")

    return None

def predict_custom(model, model_name, in_dir, out_dir, overwrite=False):
    '''
    Predict for the entire submission set.
    '''
    if out_dir is None:
        out_dir = 'models/{}/validation_samples'.format(model_name)

    test_dataloader = get_dataloader(
        in_dir=in_dir, batch_size=8,
        overwrite=overwrite, out_dir=out_dir
        )

    if not test_dataloader:
        print('Exiting...')
        sys.exit()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    print('Beginning Prediction Loop')
    tbar = tqdm(test_dataloader)
    for _, (images, targets, img_names) in enumerate(tbar):
    
        with torch.no_grad():
        # Load tensors to GPU
            images = images.to(device)
            targets = targets.to(device).squeeze(1).round().long()

        # Predict on image tensors
            outputs = model(images)[0]
            predict_imgs = [output_to_pred_imgs(output) for output in outputs]

        # Zip images, and save.
        for predict_img, img_name in zip(predict_imgs, img_names):
            image_out_path = '{}/{}'.format(out_dir, os.path.basename(img_name))
            predict_img.save(image_out_path)

    return None


def score_region(model, model_name, region, thresh=0):
    '''
    evaluate and score for a single region. save results.
    '''

    test_dataloader = get_dataloader(
        in_dir='training_data', batch_size=8, region=region
        )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    print('Beginning Prediction Loop')

    region_loss = 0

    region_ct = 0
    tbar = tqdm(test_dataloader)
    with open('region_scores.csv', 'a', newline='') as csvfile:
        csvwriter=csv.DictWriter(csvfile, fieldnames=['region', 'img_name', 'iou_score'])
        csvwriter.writeheader()
        for _, (images, masks, img_names) in enumerate(tbar):

            # Predict on image tensors
            with torch.no_grad():
                images = images.to(device)

                outputs = model(images)
                outputs = (outputs[0]>thresh).long().data
                masks = masks.to(device)

                for i, img_name in enumerate(np.array(img_names)):
                    loss = L.iou_binary(outputs[i], masks[i])
                    region_loss += loss
                    region_ct += 1
                    img_name = img_name.split('/')[-1]
                    csvwriter.writerow({'region':region, 'img_name':img_name, 'iou_score': loss})
        
        csvwriter.writerow({'region':region, 'img_name': 'avg_loss', 'iou_score': region_loss/region_ct})


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
                Default behavior checks whether images \
                exist in prediction directory.')
    test_parser.add_argument(
            '-use_lovasz', default=False, type=bool, required=False,
            help='whether to use Lovasz Hinge Loss function.')

    custom_parser = subparsers.add_parser('custom', help=predict_custom.__doc__)
    custom_parser.add_argument(
            '-in_dir', default=None, type=str, required=True,
            help='Directory of image to predict on'
            )
    custom_parser.add_argument(
            '-out_dir', default=None, type=str, required=True,
            help='Output directory'
            )
    custom_parser.add_argument(
            '-model_name', default=None, type=str, required=True,
            help='Name of model weights file in models directory')

    regional_parser = subparsers.add_parser('region', help=score_region.__doc__)
    regional_parser.add_argument(
            '-model_name', default=None, type=str, required=True,
            help='Name of model weights file in models directory')
    
    custom_args = parser.parse_args()

    if custom_args.command == 'test':
        MODEL = load_model_with_weights(model_name=custom_args.model_name, use_lovasz=custom_args.use_lovasz)
        MODEL.eval()
        predict_test_set(model=MODEL, model_name=custom_args.model_name, use_lovasz=custom_args.use_lovasz)
    
    elif custom_args.command =='custom':
        MODEL = load_model_with_weights(model_name=custom_args.model_name)
        MODEL.eval()
        predict_custom(model=MODEL, model_name=custom_args.model_name, in_dir=custom_args.in_dir, 
                out_dir=custom_args.out_dir)
    
    elif custom_args.command =='region':
        MODEL = load_model_with_weights(model_name=custom_args.model_name, use_lovasz=True, se_loss=False, aux=False)
        MODEL.eval()
        for CITY in CITY_REGIONS.keys():
            for REGION in CITY_REGIONS[CITY].keys():
                score_region(MODEL,custom_args.model_name, REGION)


    
