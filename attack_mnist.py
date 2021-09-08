#!/usr/bin/env python3

import foolbox as fb
import torch
from train_mnist import Net
import numpy as np

# Train mnist_cnn.pt.

def eval_for_model(base_name):
    model = torch.load(base_name + '_full.pt')
    model.eval()

    # Get images and labels and specify the fmodel parameter when calling the samples method.


    fmodel = fb.PyTorchModel(model, bounds=(0, 1))

    BATCH_SIZE = 16
    images, labels = fb.utils.samples(fmodel, dataset='mnist', batchsize=BATCH_SIZE)
    images = images.reshape(BATCH_SIZE, 1, 28, 28)

    attack = fb.attacks.LinfPGD()
    # Create list of epsilon values that increment on a logarithmic scale.
    epsilons = np.logspace(-1, 0, 100)

    _, advs, success = attack(fmodel, images, labels, epsilons=epsilons)

    # Merge all lists in success.

    success_list = success
    success_list = [bool(item) for sublist in success_list for item in sublist]

    success_ratio = sum(success_list)/len(success_list)
    # Calculate the success rate.
    return success_ratio


models = ['mnist_cnn', 'mnist_cnn_delta']

for model_base_name in models:
    success_ratio = eval_for_model(model_base_name)
    print(f'{model_base_name}  {success_ratio}')
