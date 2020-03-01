# ------------------------------------------
# ------------------------------------------
# Train Modified FastFCN
# ------------------------------------------
# ------------------------------------------

import fastfcn_modified as fcn_mod
import torch

def train_fastfcn_mod():

    train_dataset, train_dataloader = fcn_mod.get_dataset_and_loader()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Compile modified FastFCN model.
    model = fcn_mod.get_model()
    model.to(device)

    # Optimizer (Adam)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.005, weight_decay=0.0005)

    # Larning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=3,
                                                    gamma=0.1)

    # Loss Function (modified)
    criterion = SegmentationLosses(se_loss=args.se_loss, aux=args.aux,
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