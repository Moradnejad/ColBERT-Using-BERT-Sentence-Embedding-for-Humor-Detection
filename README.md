[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/colbert-using-bert-sentence-embedding-for/humor-detection-on-200k-short-texts-for-humor-1)](https://paperswithcode.com/sota/humor-detection-on-200k-short-texts-for-humor-1?p=colbert-using-bert-sentence-embedding-for)

## Paper Abstract

Automatic humor detection has compelling use cases in modern technologies, such as humanoid robots, chatbots, and virtual assistants. In this paper, we propose a novel approach for detecting and rating humor in short texts based on a popular linguistic theory of humor. The proposed technical method initiates by separating sentences of the given text and utilizing the BERT model to generate embeddings for each one. The embeddings are fed to a neural network as parallel lines of hidden layers in order to determine the congruity and other latent relationships between the sentences, and eventually, predict humor in the text. We accompany the paper with a novel dataset consisting of 200,000 short texts, labeled for the binary task of humor detection. In addition to evaluating our work on the novel dataset, we participated in a live machine-learning competition to rate humor in Spanish tweets. The proposed model obtained F1 scores of 0.982 and 0.869 in the performed experiments which outperform general and state-of-the-art models. The evaluation results confirm the model’s strength and robustness and suggest two important factors in achieving high accuracy in the current task: (1) usage of sentence embeddings and (2) utilizing the linguistic structure of humor in designing the proposed model.

## Dependencies

- python 3.6
- transformers package

## Results

For evaluation purposes, we created a new dataset for humor detection consisting of 200k formal short texts (100k positive and 100k negative). Experimental results show that our proposed method can determine humor in short texts with accuracy and an F1-score of 98.2 percent. Our 8-layer model with 110M parameters outperforms all baseline models with a large margin, showing the importance of utilizing linguistic structure in machine learning models.


|    Method           |    Configuration                                                                         |    Accuracy     |    Precision    |    Recall    |    F1       |
|---------------------|------------------------------------------------------------------------------------------|-----------------|-----------------|--------------|-------------|
|    XLNET        |    large             |    0.916        |    0.872        |    0.973     |    0.920    |
|    COLBERT          |             |    0.982        |    0.990       |    0.974     |    0.982    |



## Pre-trained model

If you do not want to train the model from scrach, you can download the following folder (my saved model) and put it in the same folder as your code. 

https://mega.nz/folder/MmB1gIIT#8ilUTK1-BO80aoXxKOIhpg

Then, you can use the following code to load the structure and weights of the model:

```py
import keras

model = keras.models.load_model("colbert-trained")
model.summary()
```
I uploaded a draft sample code that uses the pretrained model to simply load and predict under 2 minutes: [colbert-using-pretrained-model.ipynb](https://github.com/Moradnejad/ColBERT-Using-BERT-Sentence-Embedding-for-Humor-Detection/blob/master/colbert-using-pretrained-model.ipynb)

## How to cite
Code and dataset is released under MIT liscense.

```
Annamoradnejad, I., & Zoghi, G. (2024). ColBERT: Using BERT sentence embedding in parallel neural networks for computational humor. Expert Systems with Applications, 249, 123685.
```
2024 Paper: https://www.sciencedirect.com/science/article/abs/pii/S0957417424005517
