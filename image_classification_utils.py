import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

from collections import OrderedDict

import json

from PIL import Image
import numpy as np

from random import randint
import sys

data_dir = 'flowers'

def load_data(data_dir):
    data_sets = ['train', 'valid', 'test' ]
    dirs = {ds: data_dir + '/' + ds for ds in data_sets}

    img_rotation = 30
    img_size = 256
    img_crop_size = 224
    data_means = [0.485, 0.456, 0.406]
    data_stdevs = [0.229, 0.224, 0.225]

    dt_valid_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_crop_size),
        transforms.ToTensor(),
        transforms.Normalize(data_means, data_stdevs)
    ])

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(img_rotation),
            transforms.RandomResizedCrop(img_crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_means, data_stdevs)
        ]),
        'valid': dt_valid_test,
        'test': dt_valid_test
    }


    # Load the datasets with ImageFolder
    image_datasets = {ds: datasets.ImageFolder(dirs[ds], transform=data_transforms[ds]) for ds in data_sets }

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {ds: torch.utils.data.DataLoader(image_datasets[ds], batch_size=64, shuffle=(ds == 'train')) for ds in data_sets }

    return dataloaders, image_datasets['train'].class_to_idx

def load_categories(cat_to_name_file='cat_to_name.json'):
    with open(cat_to_name_file, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name

def create_hidden_layers(outputs, dropouts, af):

    if (len(outputs) != len(dropouts) or len(outputs) != len(af) ):
        print('ERROR: invalid hidden layers configuration:')
        print(f'Hidden layers outputs: {outputs}')
        print(f'Hidden layers dropout: {dropouts}')
        print(f'Hidden layers activation function: {af}')
        sys.exit(1)

    layers = []

    for i,output in enumerate(outputs):
        layers.append({
            'outputs' : output,
            'activation_function' : af[i],
            'dropout' : dropouts[i]
        })
    return layers

def create_output(outputs):
    return {'outputs' : outputs, 'loss_function' : nn.LogSoftmax(dim=1)}

def create_classifier(model_output, hidden_layers, output):
    layers = OrderedDict()
    layer_input = model_output
    layer_output = 0
    
    for idx, hl in enumerate(hidden_layers):
        layer_output = hl['outputs']
        layers['fc' + str(idx)] = nn.Linear(layer_input, layer_output)
        layers['af' + str(idx)] = getattr(nn, hl['activation_function'])()
        layers['do' + str(idx)] = nn.Dropout(p=hl['dropout'])
        layer_input = layer_output

    layers['fc_output'] = nn.Linear(layer_output, output['outputs'])
    layers['output'] = output['loss_function']
    
    return nn.Sequential(layers)

def create_model(dev, class_to_idx, model_type, hidden_layers, output):
    models_list = {'alexnet': 9216,
                   'densenet161': 2208,
                   'resnet18': 512,
                   'vgg16': 25088}

    try:
        model_output = models_list[model_type]
    except KeyError:
        print(f'{model_type} is not supported.')
        sys.exit(1)
        
    new_model = getattr(models, model_type)(pretrained=True)
    new_model.type = model_type
        
    for param in new_model.parameters():
        param.requires_grad = False
    
    new_model.class_to_idx = class_to_idx
    
    classifier = create_classifier(model_output, hidden_layers, output)
    if model_type == 'resnet18':
        new_model.fc = classifier
    else:
        new_model.classifier = classifier
    
    new_model.output = output
    new_model.hidden_layers = hidden_layers

    new_model.to(dev)
    
    return new_model

def create_optimizer(mod, learning_rate):
    parameters = mod.fc.parameters() if mod.type == 'resnet18' else mod.classifier.parameters()
        
    return optim.Adam(parameters, lr=learning_rate)

def do_validation(mod, testloader, criterion, dev):
    loss = 0
    accuracy = 0
    
    for images, labels in testloader:
        
        images = images.to(dev)
        labels = labels.to(dev)
        mod.to(dev)
        
        output = mod.forward(images)
        loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return loss, accuracy

def do_deep_learning(mod, optimizer, trainloader, testloader, epochs, print_interval, dev='cuda'):
    steps = 0

    mod.to(dev)
    criterion = nn.NLLLoss()

    for ep in range(epochs):
        running_loss = 0
        for data in trainloader:
            steps += 1

            images, labels = data
            images = images.to(dev)
            labels = labels.to(dev)
            
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = mod.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if print_interval > 0 and steps % print_interval == 0:
                # Training off
                mod.eval()
            
                # Gradients off
                with torch.no_grad():
                    test_loss, test_accuracy = do_validation(mod, testloader, criterion, dev)

                print("Epoch: {} of {} :".format(ep+1, epochs),
                      "Training Loss: {:.3f} ".format(running_loss/print_interval),
                      "Test Loss: {:.3f} ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(test_accuracy/len(testloader)))
                
                running_loss = 0
                
                # Training on
                mod.train()
                
# Save the checkpoint 
def save_checkpoint(mod, optimizer, epochs, learning_rate, filename):
    checkpoint = {
        'model_type': mod.type,
        'hidden_layers': mod.hidden_layers,
        'output': mod.output,
        'model_state': mod.state_dict(),
        'opt_state': optimizer.state_dict(),
        'class_to_idx': mod.class_to_idx,
        'epochs': epochs,
        'learning_rate': learning_rate
    }

    torch.save(checkpoint, filename)

# Loads a checkpoint and rebuilds the model
def load_checkpoint(filename, dev):
    # load on CPU model trained on CUDA
    checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    
    class_to_idx = checkpoint['class_to_idx']
    model_type = checkpoint['model_type']
    hidden_layers = checkpoint['hidden_layers']
    output = checkpoint['output']

    new_model = create_model(dev, class_to_idx, model_type, hidden_layers, output)
    new_model.load_state_dict(checkpoint['model_state'])
    
    learning_rate = checkpoint['learning_rate']
    optimizer = create_optimizer(new_model, learning_rate)
    optimizer.load_state_dict(checkpoint['opt_state'])
    
    epoch = checkpoint['epochs']
    
    return new_model, optimizer, epoch, learning_rate

def process_image(image_path, img_crop_size = 224):    
    data_means = np.array([0.485, 0.456, 0.406])
    data_stdevs = np.array([0.229, 0.224, 0.225])
    
    image = Image.open(image_path)
    
    # Original size
    width, height = image.size

    # Thumbnail
    if width >= height:
        image.thumbnail((img_crop_size, height * img_crop_size // width))
    else:
        image.thumbnail((width * img_crop_size // height, img_crop_size))
    
    # New size
    width, height = image.size
    
    # Crop box
    left = (width - img_crop_size)/2
    top = (height - img_crop_size)/2
    right = (width + img_crop_size)/2
    bottom = (height + img_crop_size)/2

    image = image.crop(((left, top, right, bottom)))
    
    # Convert to np.array
    image = np.array(image) / 255
    
    # Normalize
    image = (image - data_means) / data_stdevs
    
    # Make color channel first dimension
    image = image.transpose(2, 0, 1)
    
    return image 

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

# Display image with top classes
def display_classification(image_path, probs, classes, class_names):
    
    # Setup plot gird and title
    figure = plt.figure(figsize=(4, 5.4))
    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0))
    ax1.set_title(class_names[0].capitalize())
    ax2.set_title('Species')
    
    # Display image
    ax1.imshow(Image.open(image_path))
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Display predicted classes and probabilities
    y = np.arange(len(class_names))
    ax2.barh(y, probs, align='center')
    ax2.set_xlabel('Probability')
    ax2.set_yticks(y)
    ax2.set_yticklabels(class_names)

    ax2.invert_yaxis()

    figure.tight_layout()

def predict(mod, image_path, cat_to_name, topk, dev):
    img_crop_size = 224
    image = process_image(image_path, img_crop_size)
    
    torchImage = torch.from_numpy(image)
    torchImage.unsqueeze_(0)
    torchImage = torchImage.float().to(dev)
    mod.to(dev)
    output = torch.exp(mod(torchImage))
    
    probs, preds = torch.topk(output.data, topk)
    probs, preds = probs.tolist()[0], preds.tolist()[0]
    
    class_dict = {val: key for key, val in mod.class_to_idx.items()}
    classes = [class_dict[x] for x in preds] 
    class_names = [cat_to_name[x] for x in classes]
    
    return probs, classes, class_names

def print_prediction(probabilities, predictions):
    print('Predicted Categories:')
    for i, prediction in enumerate(predictions):
        print(f'{prediction} {probabilities[i]:.3f}')
 
def select_device(gpu):
    device = torch.device('cuda') if gpu and torch.cuda.is_available() else torch.device('cpu')  

    if gpu and not torch.cuda.is_available():
        print('WARNING: GPU is not available')
        device = torch.device('cpu') 

    return device