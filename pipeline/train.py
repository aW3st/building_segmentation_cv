'''
# ------------------------------------------
# ------------------------------------------
# Train Modified FastFCN
# ------------------------------------------
# ------------------------------------------
'''

import torch
import pipeline.fastfcn_modified as fcn_mod

def train_fastfcn_mod(args=None):
    '''
    Compile and train the modified FastFCN implementation.
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
        'batch_size': None, # 'input batch size for training (default: auto)'
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

    class objectView(object):
        '''
        Helper class to access dict values as attributes.

        Replaces command-line arg-parse options.
        '''
        def __init__(self, d):
            self.__dict__ = d

    # Convert dict to attribute dict
    if args is None:
        args = objectView(options)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    _, train_dataloader = fcn_mod.get_dataset_and_loader()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Compile modified FastFCN model.
    model = fcn_mod.get_model(args)
    model.to(device)

    # Optimizer (Adam)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.005, weight_decay=0.0005)

    # Larning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.1
        )

    # Loss Function (modified)
    criterion = fcn_mod.SegmentationLosses(se_loss=args.se_loss, aux=args.aux,
                                                nclass=2, 
                                                se_weight=args.se_weight,
                                                aux_weight=args.aux_weight)

    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, (image, target) in enumerate(train_dataloader, 0):
            image = image.to(device)
            target = target.to(device).squeeze(1).round().long()

            # get the inputs; data is a list of [inputs, labels]
            target.requires_grad=False
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(image)
            loss = criterion(*outputs, target)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 5 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0
                
        lr_scheduler.step()