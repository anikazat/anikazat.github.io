---
layout: post
title: "Image Captioning"
subtitle: "Build a model that will generate a short description of an image"
background: '/img/posts/image-captioning/img-capt.webp'
---

# Image Captioning

#### Project Objectives
Image captioning is a challenging problem in the deep learning domain however several research papers and Kaggle competitions have tried to address this problem in recent years and tested out various approaches. One of the key use cases of the image captioning solution would be to do a sentence-based image search or even build AI powered applications for the visually impaired.

The aim of this project is to build a model which will take images as input and generate a short description of the image. To achieve this, an image captioning model was built. This model consists of a feature extractor model (for the images) and a sequenced-based model (to generate captions).

#### Dataset
This project used the Flickr 8k dataset (“Flickr 8k Dataset,” n.d.) to train the model and to relate the images and the words for generating the captions. It contains 8,092 images which each have 5 captions describing the image.
Having multiple descriptions for each image helps deal with variation, as the same event/situation/entity can be described in various ways.