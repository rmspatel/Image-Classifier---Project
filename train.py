
# First we will import liberaries

import torch
from torch import nn
from torch import optim
from torchvision import datasets , transforms , models
import torch.nn.functional as F
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import PIL

import argparse
import math

# Functions from below

# arg_parse_  fxn will read and add arguments by command line
def arg_parse_():
    arg_parser = argparse.ArgumentParser(description='Neural Training Network')
    
    arg_parser.add_argument('--arch' , type=str , help='will choose model architecture from torchvision.models')
    arg_parser.add_argument('--save_dir' , action='store' , type=str , help='save directory must be defined either model will be lost')
    arg_parser.add_argument('--learning_rate' , type=float , default=0.001 , help='this is the learning rate for our algorithm')
    arg_parser.add_argument('--epochs' , type=int , help='Number of epochs we will use for training our neural network')
    arg_parser.add_argument('--hidden_units' , type=int , help='Number of hidden layers or unit in our neural network')
    arg_parser.add_argiment('--gpu' , action='store_true' , help='gpu will be used for calculations')
    
    args=arg_parser.parse_args()
    
    return args 
    # reference for above arg_parse_ :  https://docs.python.org/3/howto/argparse.html

# Now we will do transformation on train, valid ,test data and Load them
def train_transformer(train_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485 , 0.456 , 0.406],
                                                            [0.229 , 0.224 , 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(train_data , batch_size = 64 , shuffle=True)
    return train_data , trainloader
                            
def valid_transformer(valid_dir):
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485 , 0.456 , 0.406],
                                                            [0.229 , 0.224 , 0.225])])
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    validloader = torch.utils.data.DataLoader(valid_data , batch_size = 64)
    return valid_data , validloader

def test_transformer(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485 , 0.456 , 0.406],
                                                            [0.229 , 0.224 , 0.225])])
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    testloader = torch.utils.data.DataLoader(test_data , batch_size = 64)
    return test_data , testloader

# now we will check for gpu or cpu?
def check_for_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

# Now will defined fxn for pretrained model to download
def pretrained_model(architecture='vgg19'):
    model=models.vgg19(pretrained=True)
     # now freezing the parameters so we do't  backdrop throgh them
    for param in model.parameters():
    param.requires_grad = False
    
   return model

def classifier():
    classifier = nn.Sequential(OrderedDict([
                         ('fc1' , nn.Linear(25088 , 4096 , bias=True)),
                         ('relu1', nn.ReLU()),
                         ('dropout1' , nn.Dropout(p=0.5)),
                         ('fc2' , nn.Linear(4096,1024, bias=True)),
                         ('relu2' , nn.ReLU()),
                         ('dropout2' , nn.Dropout()),
                         ('fc3' , nn.Linear(1024,102 ,bias=True)),
                         ('output' , nn.LogSoftmax(dim=1))
                         ]))
    model.classifier = classifier
    
    return model.classifier

# fxn to train our network
def model_training(model , epochs , device , trainloader , testloader , criterion , optimizer):
    model.to(device)
    epochs = 10
    print_every=10
    steps=0
    running_loss=0
    for epoch in range(epochs):
    model.train()
        for inputs , labels in trainloader:
            steps +=1
            inputs, labels = inputs.to(device), labels.to(device)
            criterion = nn.NLLLoss()
            optimizer=optim.Adam(model.classifier.parameters() , lr=0.001)
            optimizer.zero_grad()
            logps=model.forward(inputs)
            loss=criterion(logps , labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0
                model.train() 
       
    return model

def model_validation(model , validloader , device , criterion , optimizer):
    valid_loss = 0
    accuracy = 0
    model.eval()
    for epoch , (inputs, labels) in enumerate(validloader):

            inputs, labels = inputs.to(device), labels.to(device)

            output = model.forward(inputs)
            criterion = nn.NLLLoss()
            optimizer=optim.Adam(model.classifier.parameters() , lr=0.001)
            
            valid_loss += criterion(output, labels).item()

            ps = torch.exp(output)
            top_p , top_class = ps.topk(1 , dim=1)
            equals=top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            print(f"Epoch {epoch+1}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
    return valid_loss , accuracy

def model_testing(model , testloader , device)
    test_loss = 0
    test_accuracy = 0
    model.to(device_type)
    model.eval()
    for epoch , (inputs, labels) in enumerate(testloader):

            inputs, labels = inputs.to(device_type), labels.to(device_type)

            output = model.forward(inputs)
            test_loss += criterion(output, labels).item()

            ps = torch.exp(output)
            top_p , top_class = ps.topk(1 , dim=1)
            equals=top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            print(f"Epoch {epoch+1}.. "
                      f"Test loss: {test_loss/len(testloader):.4f}.. "
                      f"Test accuracy: {test_accuracy/len(testloader):.4f}")
            if(epoch==12):
                print(f"Test accuracy of our model is : {test_accuracy * 100 /len(testloader):.4f}")
    return test_accuracy , test_loss 

def save_the_checkpoint(train_data, model, optimizer, args):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'args': args,
                  'model name': model,
                  'optimizer dict': optimizer.state_dict(),
                  'classifier':model.classifier,
                  'state_dict': model.state_dict()
                 }
    torch.save(checkpoint, 'this_checkpoint.pth')
    
# Now we define main function and call and execute all the above functions

def main():
    args=arg_parse_()
    
    # now will call data directory
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # loading datasets and dataloaders
    train_data , trainloader=train_transformer(train_dir)
    valid_data , validloader=valid_transformer(valid_dir)
    test_data , testloader=test_transformer(test_dir)
    
    # Loading our network model
    model = pretrained_model(architecture='vgg19')
    
    # checking for gpu
    device=check_for_gpu()
    
    model.to(device)
    
    # Classifier Network
    model.classifier=classifier()
    
    # Check for learnrate args
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print("use Learning rate  0.001")
    else: learning_rate = args.learning_rate
        
        
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    #Deep learning parameters
    print_every = 30
    steps = 0
    
    # Training of classifier layers using backpropogation
    model = model_training(model, epochs ,device, trainloader, testloader, criterion, optimizer)
    
    #Validate the model
    valid_loss , accuracy=model_validation(model , validloader , device , criterion , optimizer)
    
    test_loss , test_accuracy=model_testing(model , testloader , device , criterion , optimizer)
    
    # Save the checkpoint or model
    save_the_checkpoint(train_data , model,optimizer, args.save_dir)
    
# Run Program
# =============================================================================
if __name__ == '__main__': main()