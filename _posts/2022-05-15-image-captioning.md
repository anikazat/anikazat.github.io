---
layout: post
title: "Image Captioning"
subtitle: "Building a model that will generate a short description of an image"
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

```python
def caption_dictionary(token_txt):
    captions = {}
    for line in token_txt.split('\n'):
        if len(line) < 1:
            continue
        x = line.split()
        image_id = x[0].split('.')[0] # Remove caption number from image_id
        caption = ' '.join(x[1:])
        if image_id not in captions.keys():
            captions[image_id] = []
        captions[image_id].append(caption)
    return(captions)

token_txt = open("Flickr8k_text/Flickr8k.token.txt", "r").read()

caption_dict = caption_dictionary(token_txt)
```

```python
# Function to perform chosen text cleaning
def clean_captions(caption_dict):
    for key, captions in caption_dict.items():
        for i in range(len(captions)):
            caption = captions[i]
            # Text cleaning
            caption = caption.lower()
            caption = re.sub(r"[^a-zA-Z]+", " ", caption)
            caption = re.sub(" +", " ", caption)
            caption = re.sub("\s+", " ", caption)
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption

clean_captions(caption_dict)
```
I also created a function to create dictionaries for train set, testset and devset and saved them as pickle files. It will make model training easier. Next Step is to create a unique vocabulary list to store all the using captions that we just created in the previous step, and then we need to tokenize the vocabulary words. Once we have the unique tokenized words, we can create word to index and index to word dictionaries, which is very crucial for the next step: word embedding. We also need to calculate the max length of each caption to avoid out of index range errors.

```python
# Create list to contain all captions in caption_dict
caption_list = []
for key in caption_dict:
    for caption in caption_dict[key]:
        caption_list.append(caption)

# Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(caption_list)

# Get vocab_size
vocab_size = len(tokenizer.word_index) + 1

# Get max_length
max_length = max(len(caption.split()) for caption in caption_list)
```

```python
# Function to load, open, and read files
def load_file(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text

# Function used to load train/test sets 
def load_set(filename):
	txt_file = load_file(filename)
	dataset = list()
	for line in txt_file.split('\n'):
		if len(line) < 1:
			continue
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)
```

```python
text_folder_path = "/content/gdrive/MyDrive/DL_ASG_3/Flickr8k_text"
trainImages_path = text_folder_path + "/Flickr_8k.trainImages.txt"
testImages_path = text_folder_path + "/Flickr_8k.testImages.txt"

train_image_list = open(trainImages_path, 'r', encoding = 'utf-8').read().split("\n")
test_image_list =open(testImages_path, 'r', encoding = 'utf-8').read().split("\n")

train = load_set(trainImages_path)
test_test = load_set(testImages_path)
```

```python
# Function for data generator
def data_generator(dataset, caption_dict, features, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key in dataset:
            n += 1
            captions = caption_dict[key]
            for caption in captions:
                # Encode seq
                seq = tokenizer.texts_to_sequences([caption])[0]
                # Split seq
                for i in range(1, len(seq)):
                    # Input/Output split
                    in_seq, out_seq = seq[:i], seq[i]
                    # Pad input seq
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # Encode output seq
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield [X1, X2], y
                X1, X2, y = list(), list(), list()
                n = 0
```

Model Structure:
The model structure for experiment 3 is shown below in Output 1 and Figure 1. The output from the pre-trained CNN model (VGG16) is input into this model. The input shape is given as (4096,), as this is the output shape from the VGG16 model. Dropout layers were used to help avoid over fitting. The first activation function used is the Rectified Linear Unit function (ReLU), the benefits of using this function is that it helps prevent exponential growth in computation while training models. Embedding and LSTM layers were also used in this model (for the same reasons as mentioned in experiment 1 and 2). Softmax was used as the activation function on the final fully connected layer, as the model is being used for categorical predictions. Finally, the feature extractor component, sequence component and decoder component were combined.
When compiling the model, the optimiser used was adam, as it requires less memory and is efficient when working with large amounts of data. Although the models will mainly be assessed by the BLEU scores, accuracy was used when training the model as an additional way to assess the model’s performance. Since this task was a multi-class classification task, categorical cross-entropy was deemed as the most appropriate loss function.