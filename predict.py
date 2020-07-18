# This is the prediction functions file

# Importing Libraries at first
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torchvision import datasets , models , transforms
import torch.nn.functional as F

import PIL
from PIL import Image
import argparse

from train import check_for_gpu

def arg_parse_():
    # Define a parser
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--img')
    arg_parser.add_argument('--top_k' , type=int)
     # Load checkpoint created by train.py
    arg_parser.add_argument('--save_checkpoint')
    # gpu or not??
    arg_parser.add_argument('--gpu', action="store_true")

    args = parser.parse_args()
    
    return arg

def checkpoint():
    checkpoint=torch.load('this_checkpoint.pth') #loading saved checkpoint
    model=models.vgg19(pretrained = True)
    
    for param in model.parameters():
        param.requires_grad = False   # freezed feature parameters again
    
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.state_dict(checkpoint['optimizer dict'])
    model.load_state_dict(checkpoint['state dict'])
    print(model)
    return model

## process_image fxn as in part 1
def process_image(img):
 

    img = PIL.Image.open(img)
    
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    # Getting original dimensions
    original_width, original_height = img.size

    #  crop shortest side to 256
    if original_width < original_height:
        size=[256, 256**600]
    else: 
        size=[256**600, 256]
        
    img.thumbnail(size)
   
    origin = original_width, original_height
    X_neg = origin[0]-224
    Y_pos = origin[1]-224
    X_pos = origin[0]+224
    y_neg = origin[1]+224
    img = img.crop((X_neg , Y_pos , X_pos , y_neg))
    

    np_img = np.array(img)/255 

    # Normalization of each colour channel
    
    np_img = (np_img-norm_mean)/norm_std
        
    # Set the color to the first channel
    np_img = np_img.transpose(2, 0, 1)
    
    return np_img


def imshow(img, ax=None, title=None):
    #img for tensor
    if ax is None:
        fig, ax = plt.subplots()
    
    
    img = img.transpose((1, 2, 0))   
    norm_mean = np.array([0.485, 0.456, 0.406])
    norm_std = np.array([0.229, 0.224, 0.225])
    img = norm_std * img + norm_mean
    
    # to avoid noisy display of image we should kept it between 0-1
    img = np.clip(img, 0, 1)
    
    ax.imshow(img)
    
    return ax

def predict(image_path, model, top_k=10):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    
    image_path: string. Path to image, directly to image and not to folder.
    model: pytorch neural network.
    top_k: integer. The top K classes to be calculated
    
    returns top_probabilities(k), top_labels
    '''
    
    # seems like GPU is not required for this part so we can change our model to CPU only
    model.to("cpu")
    
    model.eval()
    # Convert image from numpy to torch
    torch_img = torch.from_numpy(np.expand_dims(process_image(image_path), 
                                                  axis=0)).type(torch.FloatTensor).to("cpu")

    log_ps = model.forward(torch_img)  # OUTPUT OF ACTIVATION FXN

    lin_ps = torch.exp(log_ps)

    # SO WE CAN FIND top k results
    head_ps, head_class = lin_ps.topk(top_k)
    
    # Detatch all of the details
    head_ps = np.array(top_ps.remove())[0]  
    head_class = np.array(top_class.remove())[0]
    
    idx = {i:j for i, j in model.class_to_idx.items()}
    head_class = [idx[k] for k in head_class]
    head_flowers = [name[p] for p in head_class]
    
    return head_ps, head_class, head_flowers


def displaying_probs():
    image_path = "flowers/test/9/image_06413.jpg"

    # setting plot description
    plt.figure(figsize = (8,12))
    ax = plt.subplot(2,1,1)

    # Set up title
    flower_number = image_path.split('/')[2]
    title_ = name[flower_number]

    # Plot flower
    img = process_image(image_path)
    imshow(img, ax, title = title_);

    # Make prediction
    probs, classes , flowers = predict(image_path, model) 

    # Plot bar chart
    plt.subplot(2,1,2)
    sns.barplot(x=probs, y=flowers);
    plt.show()
        
def main():

    
    
    args = arg_parser()
    
    # Load categories to names json file from part 1
    import json
    with open(args.category_names, 'r') as f:
        	  name = json.load(f)

    # Loading trained model from train.py
    model = checkpoint(args.save_checkpoint)
    
    # image preprocessing
    image_tensor = process_image(args.img)
    
    # Checking for gpu here
    device = check_gpu();
    head_probs, head_labels, head_flowers = predict(image_tensor, model, device, name, args.top_k)
    
    #Getting probabilities
    displaying_probs()

# =============================================================================
# Run Program
# =============================================================================
if __name__ == '__main__': main()        