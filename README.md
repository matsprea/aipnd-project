
# Developing an AI application

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 

<img src='assets/Flowers.png' width=500px>

The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

We'll lead you through each part which you'll implement in Python.

When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.

First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.


```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Load the data

Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.

The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.

The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
 


```python
data_dir = 'flowers'
data_sets = ['train', 'valid', 'test' ]

dirs = {ds: data_dir + '/' + ds for ds in data_sets}

# dirs
```


```python
# Define your transforms for the training, validation, and testing sets

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
]);

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

# data_transforms
```


```python
# Load the datasets with ImageFolder
image_datasets = {ds: datasets.ImageFolder(dirs[ds], transform=data_transforms[ds]) for ds in data_sets }

# image_datasets
```


```python
# Using the image datasets and the trainforms, define the dataloaders
dataloaders = {ds: torch.utils.data.DataLoader(image_datasets[ds], batch_size=64, shuffle=(ds == 'train')) for ds in data_sets }

# dataloaders
```

# Label mapping

You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.


```python
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# cat_to_name
```

# Building and training the classifier

Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.

We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:

* Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
* Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
* Train the classifier layers using backpropagation using the pre-trained network to get the features
* Track the loss and accuracy on the validation set to determine the best hyperparameters

We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!

When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.

One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to
GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.

**Note for Workspace users:** If your network is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. Typically this happens with wide dense layers after the convolutional layers. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.


```python
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
```


```python
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
```


```python
def create_output(outputs):
    return {'outputs' : outputs, 'loss_function' : nn.LogSoftmax(dim=1)}
```


```python
def create_model(dev, class_to_idx, model_type, hidden_layers, output):
    models_list = { 'vgg16': 25088, 'alexnet': 9216, 'resnet18': 512 }

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
```


```python
def create_optimizer(mod, learning_rate):
    parameters = mod.fc.parameters() if mod.type == 'resnet18' else mod.classifier.parameters()
        
    return optim.Adam(parameters, lr=learning_rate)
```


```python
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
```


```python
def do_deep_learning(mod, criterion, optimizer, trainloader, testloader, epochs, print_interval, dev='cuda'):
    steps = 0

    mod.to(dev)

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

            if steps % print_interval == 0:
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
                
```


```python
layers = create_hidden_layers([2048, 512], [0.5, 0.5], ['ReLU', 'ReLU'])
output = create_output(102)

model_type = 'vgg16'
class_to_idx = image_datasets['train'].class_to_idx
model = create_model(device, class_to_idx, model_type, layers, output)

criterion = nn.NLLLoss()

learning_rate = 1e-3
optimizer = create_optimizer(model, learning_rate)

epochs = 16
print_interval = 50
```

    Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /root/.torch/models/vgg16-397923af.pth
    100%|██████████| 553433881/553433881 [00:04<00:00, 117561840.33it/s]



```python
do_deep_learning(model, criterion, optimizer, dataloaders['train'], dataloaders['valid'], epochs, print_interval, device)
```

    Epoch: 1 of 16 : Training Loss: 2.983  Test Loss: 1.763  Test Accuracy: 0.578
    Epoch: 1 of 16 : Training Loss: 2.498  Test Loss: 1.279  Test Accuracy: 0.666
    Epoch: 2 of 16 : Training Loss: 1.952  Test Loss: 0.999  Test Accuracy: 0.732
    Epoch: 2 of 16 : Training Loss: 1.964  Test Loss: 0.877  Test Accuracy: 0.774
    Epoch: 3 of 16 : Training Loss: 1.556  Test Loss: 0.790  Test Accuracy: 0.781
    Epoch: 3 of 16 : Training Loss: 1.746  Test Loss: 0.740  Test Accuracy: 0.809
    Epoch: 4 of 16 : Training Loss: 1.342  Test Loss: 0.723  Test Accuracy: 0.806
    Epoch: 4 of 16 : Training Loss: 1.601  Test Loss: 0.637  Test Accuracy: 0.813
    Epoch: 5 of 16 : Training Loss: 1.138  Test Loss: 0.582  Test Accuracy: 0.858
    Epoch: 5 of 16 : Training Loss: 1.493  Test Loss: 0.599  Test Accuracy: 0.839
    Epoch: 6 of 16 : Training Loss: 1.044  Test Loss: 0.562  Test Accuracy: 0.845
    Epoch: 6 of 16 : Training Loss: 1.507  Test Loss: 0.548  Test Accuracy: 0.847
    Epoch: 7 of 16 : Training Loss: 0.889  Test Loss: 0.521  Test Accuracy: 0.858
    Epoch: 7 of 16 : Training Loss: 1.411  Test Loss: 0.510  Test Accuracy: 0.857
    Epoch: 8 of 16 : Training Loss: 0.819  Test Loss: 0.571  Test Accuracy: 0.854
    Epoch: 8 of 16 : Training Loss: 1.327  Test Loss: 0.494  Test Accuracy: 0.889
    Epoch: 9 of 16 : Training Loss: 0.730  Test Loss: 0.517  Test Accuracy: 0.873
    Epoch: 9 of 16 : Training Loss: 1.338  Test Loss: 0.500  Test Accuracy: 0.864
    Epoch: 10 of 16 : Training Loss: 0.579  Test Loss: 0.507  Test Accuracy: 0.859
    Epoch: 10 of 16 : Training Loss: 1.407  Test Loss: 0.465  Test Accuracy: 0.880
    Epoch: 11 of 16 : Training Loss: 0.509  Test Loss: 0.467  Test Accuracy: 0.883
    Epoch: 11 of 16 : Training Loss: 1.255  Test Loss: 0.482  Test Accuracy: 0.881
    Epoch: 12 of 16 : Training Loss: 0.435  Test Loss: 0.478  Test Accuracy: 0.870
    Epoch: 12 of 16 : Training Loss: 1.223  Test Loss: 0.462  Test Accuracy: 0.887
    Epoch: 13 of 16 : Training Loss: 0.344  Test Loss: 0.460  Test Accuracy: 0.866
    Epoch: 13 of 16 : Training Loss: 1.265  Test Loss: 0.461  Test Accuracy: 0.877
    Epoch: 14 of 16 : Training Loss: 0.277  Test Loss: 0.477  Test Accuracy: 0.880
    Epoch: 14 of 16 : Training Loss: 1.149  Test Loss: 0.474  Test Accuracy: 0.876
    Epoch: 15 of 16 : Training Loss: 0.179  Test Loss: 0.445  Test Accuracy: 0.877
    Epoch: 15 of 16 : Training Loss: 1.227  Test Loss: 0.495  Test Accuracy: 0.870
    Epoch: 16 of 16 : Training Loss: 0.147  Test Loss: 0.490  Test Accuracy: 0.877
    Epoch: 16 of 16 : Training Loss: 1.209  Test Loss: 0.434  Test Accuracy: 0.882


## Testing your network

It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.


```python
def check_accuracy_on_test(mod, testloader, dev):    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            
            images = images.to(dev)
            labels = labels.to(dev)
            mod.to(dev)
            
            outputs = mod(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on the %d test images: %d %%' % (total, 100 * correct / total))
    
check_accuracy_on_test(model, dataloaders['test'], device)
```

    Accuracy on the 819 test images: 76 %


## Save the checkpoint

Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.

```model.class_to_idx = image_datasets['train'].class_to_idx```

Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.


```python
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

save_checkpoint(model, optimizer, epochs, learning_rate, 'checkpoint.pth')
```

## Loading the checkpoint

At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.


```python
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
    
model, optimizer, epochs, learning_rate = load_checkpoint('checkpoint.pth', device)
```

# Inference for classification

Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 

```python
probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
```

First you'll need to handle processing the input image such that it can be used in your network. 

## Image Preprocessing

You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 

First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.

Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.

As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 

And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.


```python
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
```

To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).


```python
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
```

## Class Prediction

Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.

To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.

Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.

```python
probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
```


```python
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
```


```python
def predict(dev, mod, image_path, img_crop_size, topk=5):
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

img_path = 'flowers/valid/102/image_08038.jpg'
probs, classes, class_names = predict(device, model, img_path, img_crop_size) 
display_classification(img_path, probs, classes, class_names) 

```


![png](assets/output_31_0.png)


## Sanity Checking

Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:

<img src='assets/inference_example.png' width=300px>

You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.


```python
# Select a random image
def get_random_image_path(cat, data_dir='flowers/valid'):
    valid_categories = os.listdir(data_dir)
    random_category = randint(0, len(valid_categories) - 1)
    random_category = valid_categories[random_category]
    
    images_in_category = os.listdir(data_dir + '/' + random_category)
    random_image = randint(0, len(images_in_category) - 1)
    random_image = images_in_category[random_image]
    
    path = data_dir + '/' + random_category + '/' + random_image
    return path
```


```python
# Predict a radom image
def predict_random_image(mod, cat, topk=5, dev='cuda', img_crop_size=224):
    random_image_path = get_random_image_path(cat)
    
    probs, classes, class_names = predict(dev, mod, random_image_path, img_crop_size, topk) 

    display_classification(random_image_path, probs, classes, class_names) 
```


```python
predict_random_image(model, cat_to_name, 5, device, 244)
```


![png](assets/output_35_0.png)



```python

```
