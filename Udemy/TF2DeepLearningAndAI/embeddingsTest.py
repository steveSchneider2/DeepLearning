# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 07:48:27 2021

@author: steve
"""
import wget, os
#  wget -N http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz -P data/
url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz'
url2 = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz'
output ='d:/data/udemy/hadelin'
dwnldFile = wget.download(url, out=output)  # this worked! :)
dwnldFile2 = wget.download(url2, out=output)  # this worked! :)
#dwnldFile = '10_Food_Classes_All_Data'
destination = os.path.join(output, dwnldFile)  # this join gives two dif slashes :(
destination = os.path.join(output, dwnldFile)  # this join gives two dif slashes :(
source = 'C:/Users/steve/Downloads/10_Food_classes_all_data.zip'
labels
