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

#### Models
To solve the image captioning problem, I experimented on a combination of 2 models:
* Image Encoder (feature extractor for the images)
* Language Model (sequence model to generate captions)

Image Features Extraction:
The network built is a combination of encoder and decoder, with CNN being used as the encoding layer. The CNN layer extracts the features from the images and is connected to a Long short-term memory (LSTM) network, a type of Recurrent Neural Network (RNN). I tested out the VGG16 image encoding model in this project.
VGG16 requires the images to be converted to 224x224 size, the images are then converted to numpy arrays ready for pre-processing, and the top layers of the model were removed.
```python
# Load the pre-trained CNN model
model = VGG16()

# Remove the prediction layer because we will add our own
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

# View model summary
model.summary()
```
A loop function is created to call the image features extraction function to loop over all the images, then all the encoded train/test/dev images are saved as pickle files.
```python
# Extract features from the images in Flicker8k_Dataset folder and add to dictionary
features = {}
directory = os.path.join(DIR, 'Flicker8k_Dataset')

for img_name in tqdm(os.listdir(directory)):
    img_path = directory + '/' + img_name
    # Shape required by vgg16 is (224, 224)
    image = load_img(img_path, target_size=(224, 224))
    # Convert numpy array
    image = img_to_array(image)
    # Reshape data (1, 224, 224, 3)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # Use vgg preprocess_input function
    image = preprocess_input(image)
    # Extract features from model.predict
    feature = model.predict(image, verbose=0)
    image_id = img_name.split('.')[0]
    features[image_id] = feature

# Save features so we don't need to re-do this long process next time
pickle.dump(features, open(os.path.join(DIR, 'vgg16_features.pkl'), 'wb'))

features = pickle.load(open(os.path.join(DIR, 'vgg16_features.pkl'), 'rb'))
```
Pre-processing text/Captions:
Text captions also need to be cleaned and pre-processed before training the model. First, we need to create a dictionary to store all the image names as keys and captions as values. Once that’s done, we can cleanse the captions by converting them to lower texts, removing extra whitespaces and removing symbols etc. We also need to add the ‘Start seq’ and ‘End Seq’ on each line.
