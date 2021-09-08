#!/usr/bin/env python3

'''
This short and simple script downloads the imagnette dataset.
'''

import urllib.request
import os

# The URL to the imagenette dataset
url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"

# Download the dataset
urllib.request.urlretrieve(url, "imagenette2.tgz")

# Extract the dataset
os.system("tar -xvzf imagenette2.tgz")

