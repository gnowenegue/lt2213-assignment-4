# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: LT2213 Assignment 4
#     language: python
#     name: lt2213-assignment-4
# ---

# %% [markdown]
# # LSTMs and Transformers for Word Sense Disambiguation
#
# **Nikolai Ilinykh, Adam Ek, Simon Dobnik, and others. Last modified by Felix Morger.**
#
# The lab is an exploration and learning exercise to be done in a group and also in discussion with the teachers and other students.
#
# Before starting, please read the instructions on how to work in groups on Canvas.
#
# Write all your answers and the code in the appropriate boxes below.
#
# Static distributional vectors are not trained to distinguish between different *word senses*. They learn all senses per token or word. We continue our exploration of word vectors by considering *trainable vectors* or *word embeddings* for Word Sense Disambiguation (WSD) that include semantic information from the contexts in which a word occurs. We experiment with and compare LSTMs and transformer models, e.g. BERT. The purpose of this exercise is also to learn how to use vector representations from neural models in a downstream task of word sense disambiguation.
#
# **Dependencies**
#
# * Pytorch
#     * Installation instructions: https://pytorch.org/
#     * Tutorials: https://pytorch.org/tutorials/beginner/basics/intro.html
#     * Some useful basic operations: https://jhui.github.io/2018/02/09/PyTorch-Basic-operations
# * BERT
#     * [HuggingFace transformers](https://huggingface.co/docs/transformers/en/index)
#     * [BERT model (example)](https://huggingface.co/google-bert/bert-base-uncased)
#
# **Running the code**
#
# As we are learning about the models, and also what methods work and do not work for our semantic tasks, we are not interested in achieving a state-of-the-art performance. We are learning about different implementations and differences in performance in different conditions.
#
# **On using generative AI for this assignment:** For this lab, the use of generative AI is permitted as a supporting tool, provided it is done in a responsible and conscious manner and that you state clearly with each question how it was used. However, generative AI must never replace the intellectual work you are expected to carry out. Note that the purpose of this lab is to learn some basic coding of the main neural architectures used in natural language processing. You are responsible for ensuring that such tools are used in a way that supports the development of the skills the module is designed to promote. It is your responsibility to ensure that submitted work is the result of independent intellectual effort.
#
# **Getting help:** We encourage you to use Canvas discussions to post questions and interact with teachers and also each other. Provide useful tips, but of course do not reveal the exact answer across the groups as each group should should work out their own solutions. Remember that in most cases there is also not a single correct answer and implementations may differ.
#
#
# ## Word Sense Disambiguation Task
#
# The goal of word sense disambiguation is to train a model to find the sense of a word (homonyms of a word-form). For example, the word "bank" can mean "sloping land" or "financial institution". 
#
# (a) "I deposited my money in the **bank**" (financial institution)
#
# (b) "I swam from the river **bank**" (sloping land)
#
# In case a) and b), we can determine the meaning of "bank" based on the *context*. To utilize context in a semantic model, we use *contextualized word representations*.
#
# Previously, we worked with *static word representations*, i.e., the representation does not depend on the context. To illustrate, we can consider sentences (a) and (b), where the word **bank** would have the same static representation in both sentences, which means that it becomes difficult for us to predict its sense. What we want is to create representations that depend on the context, i.e., *contextualized embeddings*.
#
# As we have discussed in the class, contextualized representations can come in the form of pre-training the model for some "general" task and then fine-tuning it for some downstream task. Here we will do the following:
#
# (1) Train and test LSTM model directly for word sense disambiguation. We will learn contextualized representations within this model.
#
# (2) Take BERT that was pre-trained on masked language modeling and next sentence prediction. Fine-tune it on our data and test it for the word sense disambiguation on the task dataset. The idea for you is to explore how pre-trained contextualized representations from BERT can be updated and used for the downstream task of word sense disambiguation.
#
# Your overall task in this lab is to create a neural network model that can disambiguate the word sense of 30 different words.

# %%
# install dependencies
# %pip install -r ./packages.txt

# %%
# first we import some packages that we need

# here add any package that you will need later

import torch
import torch.nn as nn

import random
import csv
import os
from collections import Counter, defaultdict

cuda_available = torch.cuda.is_available()
mps_available = torch.backends.mps.is_available()

print(f"CUDA (NVIDIA GPU) Available: {cuda_available}")
print(f"MPS (Apple Silicon GPU) Available: {mps_available}")

if cuda_available:
    device = torch.device('cuda')
    decision = "NVIDIA GPU (CUDA)"
elif mps_available:
    device = torch.device('mps')
    decision = "Apple Silicon GPU (MPS)"
else:
    device = torch.device('cpu')
    decision = "CPU"

print(f"Using device: {decision}")

# our hyperparameters (add more when/if you need them)
# device = torch.device('cuda:0')

batch_size = 8
learning_rate = 0.001
epochs = 3


# %% [markdown]
# # 1. Working with Data
#
# A central part of any machine learning system is the data we're working with.
#
# In this section, we will split the data (the dataset is in `wsd_data.txt`) into a training set and a test set.
#

# %% [markdown]
# ## Data
#
# The dataset we will use contains different word senses for 30 different words. The data is organized as follows (values separated by tabs), where each line is a separate item in the dataset:
#
# - Column 1: word-sense, e.g., keep%2:42:07::
# - Column 2: word-form, e.g., keep.v
# - Column 3: index of word, e.g., 15
# - Column 4: white-space tokenized context, e.g., Action by the Committee In pursuance of its mandate , the Committee will continue to keep under review the situation relating to the question of Palestine and participate in relevant meetings of the General Assembly and the Security Council . The Committee will also continue to monitor the situation on the ground and draw the attention of the international community to urgent developments in the Occupied Palestinian Territory , including East Jerusalem , requiring international action .
#

# %% [markdown]
# ### Splitting the Data
#
# Your first task is to separate the data into a *training set* and a *test set*.
#
# The training set should contain 80% of the examples, and the test set the remaining 20%.
#
# The examples for the test/training set should be selected **randomly**.
#
# Save each dataset into a .csv file for loading later.
#
# **[2 marks]**

# %%
def data_split(dataset_path):
    # your code goes here

    with open(dataset_path, 'r', encoding='utf-8') as f:
        lines = [line.strip().split('\t') for line in f if line.strip()]

    # shuffle the data randomly
    random.seed(44)
    random.shuffle(lines)

    # split the data into 80% train and 20% test
    split_idx = int(len(lines) * 0.8)
    train_split = lines[:split_idx]
    test_split = lines[split_idx:]

    # save the splits to .csv files
    def save_csv(data, filename):

        # ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)

    save_csv(train_split, './data/train.csv')
    save_csv(test_split, './data/test.csv')

    return train_split, test_split


# %%
train_data, test_data = data_split('./wsd_data.txt')

print(f"Total data size: {len(train_data) + len(test_data)}")
print(f"Training data size: {len(train_data)}")
print(f"Testing data size: {len(test_data)}")


# %% [markdown]
# ### Creating a Baseline
#
# Your second task is to create a *baseline* for the task.
#
# A baseline is a "reality check" for a model. Given a very simple heuristic/algorithmic/model solution to the problem, can our neural network perform better than this? Baselines are important as they give us a point of comparison for the actual models. They are commonly used in NLP. Sometimes baseline models are not simple models but previous state-of-the-art.
#
# In this exercise, we will have a simple baseline model that is the "most common sense" (MCS) baseline. For each word form, find the most commonly assigned sense to the word and label a word with that sense. In a fictional dataset, "bank" has two senses: "financial institution," which occurs 5 times, and "side of the river," which occurs 3 times. Thus, all 8 occurrences of "bank" are labeled "financial institution," yielding an MCS accuracy of 5/8 = 62.5%. If a model obtains a higher score than this, we can conclude that the model *at least* is better than selecting the most frequent word sense.
#
# Your task is to write the code for this baseline, train, and test it. The baseline has the knowledge about labels and their frequency only from the train data. You evaluate it on the test data by comparing the ground-truth sense with the one that the model predicts. A good "dumb" baseline in this case is the one that performs quite badly. Expect the model to perform around 0.30 in terms of accuracy. You should use accuracy as your main metric; you can also compute the F1-score.
#
# **[2 marks]**
#

# %%
def mcs_baseline(train_data, test_data):

    # your code goes here
    """
    most common sense (MCS) baseline.
    1. finds the most frequent sense for each word-form in the training set.
    2. uses that sense to predict labels for the test set.
    """
    # dictionary to store sense frequencies for each word form
    # key: word form (e.g., 'bank.n'), value: counter of word senses
    frequencies = defaultdict(Counter)

    # step 1: "train" - count word sense occurrences in the training data
    for row in train_data:
        train_word_sense = row[0]   # column 1: word sense
        train_word_form = row[1]    # column 2: word form

        frequencies[train_word_form][train_word_sense] += 1

    # step 2: "test" - evaluate accuracy on the testing data
    correct_predictions = 0
    total_examples = len(test_data)

    for row in test_data:
        test_word_sense = row[0]
        test_word_form = row[1]

        counts = frequencies.get(test_word_form)

        # predict the MCS for this word form
        prediction = max(
            counts,
            key=counts.get,
            default=None
        ) if counts else None

        if prediction == test_word_sense:
            correct_predictions += 1

    # calculate and return accuracy
    accuracy = correct_predictions / total_examples if total_examples > 0 else 0
    return accuracy


# %%
# calculate and print the baseline accuracy
baseline_accuracy = mcs_baseline(train_data, test_data)
print(f"MCS Baseline Accuracy: {baseline_accuracy:.4f}")


# %% [markdown]
# ### Creating Data Iterators
#
# To train a neural network, we first need to prepare the data. This involves converting words (and labels) to a number and organizing the data into batches. We also want the ability to shuffle the examples such that they appear in a random order.
#
# Your task is to create a dataloader for the training and test set you created previously.
#
# You are encouraged to adjust your own dataloader you built for previous assignments. Some things to take into account:
#
# 1. Tokenize inputs, keep a dictionary of word-to-IDs and IDs-to-words (vocabulary), fix paddings. You might need to consider doing these for each of the four fields in the dataset.
# 2. Your dataloader probably has a function to process data. Process each column in the dataset.
# 3. You might want to clean the data a bit. For example, the first column has some symbols, which might be unnecessary. It is up to you whether you want to remove them and clean this column or keep labels the way they are. In any case, you must provide an explanation of your decision and how you think it will affect the performance of your model. Data and its preprocessing matters, so motivate your decisions.
# 4. Organize your dataset into batches and shuffle them. You should have something akin to data iterators so that your model can take them.
#
# Implement the dataloader and perform necessary preprocessings.
#
# [**2 marks**]

# %%
def dataloader(path):

    # your code goes here
    # below are only some examples!
    
    def __getitem__(self, idx):
        
        return Y

    def __len__(self):
        
        return X
    
def data_load(something):
    
    return dataloader_train, dataloader_test


# %% [markdown]
# # 2.1 LSTM for Word Sense Disambiguation
#
# In this section, we will train an LSTM model to predict word senses based on *contextualized representations*.
#
# You can read more about LSTMs [here](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).
#

# %% [markdown]
# ### Model
#
# We will use a **bidirectional** Long Short-Term Memory (LSTM) network to create a representation for the sentences and a **linear** classifier to predict the sense of each word.
#
# As we discussed in the lecture, bidirectional LSTM is using **two** hidden states: one that goes in the left-to-right direction, and another one that goes in the right-to-left direction. PyTorch documentation on LSTMs can be found [here](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html). It says that if the bidirectional parameter is set to True, then "h_n will contain a concatenation of the final forward and reverse hidden states, respectively." Keep it in mind because you will have to ensure that your linear layer for prediction takes input of that size.
#
# When we initialize the model, we need a few things:
#
# 1) An embedding layer: a dictionary from which we can obtain word embeddings
# 2) A LSTM-module to obtain contextual representations
# 3) A classifier that computes scores for each word-sense given *some* input
#
# The general procedure is the following:
#
# 1) For each word in the sentence, obtain word embeddings
# 2) Run the embedded sentences through the LSTM
# 3) Select the appropriate hidden state
# 4) Predict the word-sense 
#
# **Suggestion for efficiency:** *Use a low dimensionality (32) for word embeddings and the LSTM when developing and testing the code, then scale up when running the full training/tests*
#
# Your tasks will be to create **two different models** (both follow the two outlines described above).

# %% [markdown]
# -----

# %% [markdown]
# Your first model should make a prediction from the LSTM's representation of the target word.
#
# In particular, you run your LSTM on the context in which the target word is used. LSTM will produce a sequence of hidden states. Each hidden state corresponds to a single word from the input context. For example, you should be able to get 37 hidden states for a context that has 37 words/elements in it. Next, take the LSTM's representation of the target word. For example, it can be hidden state number 5, because the fifth word in your context is the target word that you want to predict the meaning for. This target's word representation is the input to your linear layer that makes the final prediction.
#
# **[5 marks]**

# %%
class WSDModel_approach1(nn.Module):
    def __init__(self, ...):
        
        # your code goes here
        self.embeddings = ...
        self.rnn = ...
        self.classifier = ...
    
    def forward(self, batch):
        # your code goes here
        
        return predictions


# %% [markdown]
# Your second model should make a prediction from the final hidden state of your LSTM.
#
# In particular, do the same first steps as in the first approach. But then to make a prediction with your linear layer, you will need to take the last hidden state that your LSTM produces for the whole sequence.
#
# **[5 marks]**

# %%
class WSDModel_approach2(nn.Module):
    def __init__(self, ...):
        # your code goes here
    
    def forward(self, ...):
        # your code goes here
        
        return predictions


# %% [markdown]
# ### Training and Testing the Model
#
# Now we are ready to train and test our model. What we need now is a loss function, an optimizer, and our data. 
#
# - First, create the loss function and the optimizer.
# - Next, iterate over the number of epochs (i.e., how many times we let the model see our data). 
# - For each epoch, iterate over the dataset to obtain batches. Use the batch as input to the model, and let the model output scores for the different word senses.
# - For each model output, calculate the loss (and print the loss) on the output and update the model parameters.
# - Reset the gradients and repeat.
# - After all epochs are done, test your trained model on the test set and calculate the total and per-word-form accuracy of your model.
#
# Implement the training and testing of the model.
#
# **[4 marks]**
#
# **Suggestion for efficiency:** *When developing your model, try training and testing the model on one or two batches (for each epoch) of data to make sure everything works! It's very annoying if you train for N epochs to find out that something went wrong when testing the model, or to find that something goes wrong when moving from epoch 0 to epoch 1.*
#
# Do not forget to save your best models as .pickle files. The results should be reproducible for us to evaluate your models.
#

# %%
train_iter, test_iter, vocab, labels = dataloader(path_to_folder)

loss_function = ...
optimizer = ...
model = ...

for _ in range(epochs):
    # train model
    ...
    
# test model after all epochs are completed

# %% [markdown]
# # 2.2 Fine-tuning and Testing BERT for Word Sense Disambiguation

# %% [markdown]
# In this section of the lab, you'll try out the transformer, specifically the BERT model. For this, we'll use the Hugging Face library ([https://huggingface.co/](https://huggingface.co/)).
#
# You can find the documentation for the BERT model [here](https://huggingface.co/transformers/model_doc/bert.html) and a general usage guide [here](https://huggingface.co/docs/transformers/quicktour).
#
# What we're going to do is *fine-tune* the BERT model, i.e., update the weights of a pre-trained model. That is, we have a model that is pre-trained on masked language modeling and next sentence prediction (kind of basic, general tasks which are useful for a lot of more specific tasks), but now we apply it to word sense disambiguation with the word representations it has learned.
#
# We'll use the same data splits for training and testing as before, but this time you will use a different dataloader.
#
# Now you create an iterator that collects N sentences (where N is the batch size) then use the BertTokenizer to transform the sentence into integers. For your dataloader, remember to:
# * Shuffle the data in each batch
# * Make sure you get a new iterator for each *epoch*
# * Create a vocabulary of *sense-labels* so you can calculate accuracy 
#
# We then pass this batch into the BERT model (you must have pre-loaded its weights) and update the weights (fine-tune). The BERT model will encode the sentence, then we send this encoded sentence into a prediction layer and collect what it outputs.
#
# As input to the prediction layer, you are free to play with different types of information. For example, the expected way would be to use CLS representation. You can also use other representations and compare them.
#
# About the hyperparameters and training:
# * For BERT, usually a lower learning rate works best, between 0.0001-0.000001.
# * BERT takes a lot of resources, running it on CPU will take ages, utilize the GPUs :)
# * Since BERT takes a lot of resources, use a small batch size (4-8)
# * Computing the BERT representation, make sure you pass the mask. It tells the model to ignore padded tokens when computing attention.
#
# **[12 marks]**

# %%
def dataloader_for_bert(path_to_file, batch_size):
    ...


# %%
class BERT_WSD(nn.Module):
    def __init__(self, ...):
        # your code goes here
        self.bert = ...
        self.classifier = ...
    
    def forward(self, batch):
        # your code goes here
        
        return predictions


# %%
loss_function = ...
optimizer = ...
model = ...

for _ in range(epochs):
    # train model
    ...
    
# test model after all epochs are completed

# %% [markdown]
# # 3. Evaluation

# %% [markdown]
# Explain the difference between the two LSTMs that you have implemented for word sense disambiguation.
#
# Important note: your LSTMs should be nearly the same, but your linear layer must take different inputs. Describe why and how you think this difference will affect the performance of different LSTMs. How does the contextual representation of the whole sequence perform? How does the representation of the target word perform? What is better and for what situations? Why do we observe these differences?
#
# What kind of representations are the different approaches using to predict word senses?
#
# **[4 marks]**

# %%

# %% [markdown]
# Evaluate your model with per-word form *accuracy* and comment on the results you get. How does the model perform in comparison to the baseline, and how do the models compare to each other? 
#
# Expand on the evaluation by sorting the word-forms by the number of senses they have. Are word forms with fewer senses easier to predict? Give a short explanation of the results you get based on the number of senses per word.
#
# **[4 marks]**

# %%

# %% [markdown]
# How do the LSTMs perform in comparison to BERT? What's the difference between representations obtained by the LSTMs and BERT?
#
# **[4 marks]**

# %%

# %% [markdown]
# What could we do to improve all WSD models that we have worked with in this assignment?
#
# **[2 marks]**

# %%

# %% [markdown]
# # Readings
#
# [1] Kågebäck, M., & Salomonsson, H. (2016). Word Sense Disambiguation using a Bidirectional LSTM. arXiv preprint arXiv:1606.03568.
#
# [2] On WSD: https://web.stanford.edu/~jurafsky/slp3/slides/Chapter18.wsd.pdf

# %% [markdown]
# ## Your reflections on this lab
#
# Write below your general thoughts, experiences, or reflections on how you worked on this lab.

# %%

# %% [markdown]
# ## Statement of contribution
#
# Briefly state how many times you have met for discussions, who was present, to what degree each member contributed to the discussion and the final answers you are submitting.

# %%

# %% [markdown]
# ## Marks
#
# This assignment has a total of 46 marks.
