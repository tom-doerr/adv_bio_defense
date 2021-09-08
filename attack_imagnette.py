#!/usr/bin/env python3

import foolbox as fb
import torch
import numpy as np
from train_imagenette_3 import Net, test_data, train_data, test_data_loader

import itertools

# Train mnist_cnn.pt.


def eval_for_model(base_name):
    model = torch.load(base_name + '_full.pt')
    model.eval()

    # Get images and labels and specify the fmodel parameter when calling the samples method.

    fmodel = fb.PyTorchModel(model, bounds=(0, 1))

    # Run attack on model.

    attack = fb.attacks.LinfPGD()
    success = 0
    total = 0

    success_list  = []
    for i, (images, labels) in enumerate(test_data_loader):
        if i % 10 == 0:
            print(f'{i}/{len(test_data)}')

        images = images.cuda()[:32]
        labels = labels.cuda()[:32]

        total += len(images)

        epsilons = np.logspace(-2, 0, 100)
        _, advs, success = attack(fmodel, images, labels, epsilons=epsilons)
        success_list += list(itertools.chain.from_iterable(success))
        break

    success = sum(success_list)
    total = len(success_list)
    return success / total




models = ['models/model', 'models/model_delta']

for model_base_name in models:
    success_ratio = eval_for_model(model_base_name)
    print(f'{model_base_name}  {success_ratio}')
