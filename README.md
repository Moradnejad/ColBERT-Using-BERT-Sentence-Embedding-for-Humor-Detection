[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/colbert-using-bert-sentence-embedding-for/humor-detection-on-200k-short-texts-for-humor-1)](https://paperswithcode.com/sota/humor-detection-on-200k-short-texts-for-humor-1?p=colbert-using-bert-sentence-embedding-for)

## Introduction

Automatic humor detection has interesting use cases in modern technologies, such as chatbots and personal assistants. In this paper, we describe a novel approach for detecting humor in short texts using BERT sentence embedding. Our proposed model uses BERT to generate sentence embeddings for texts which are sent as input to a neural network that predicts the target value. 

## Dependencies

- python 3.6
- transformers package

## Results

For evaluation purposes, we created a new dataset for humor detection consisting of 200k formal short texts (100k positive and 100k negative). Experimental results show that our proposed method can determine humor in short texts with accuracy and an F1-score of 98.2 percent. Our 8-layer model with 110M parameters outperforms all baseline models with a large margin, showing the importance of utilizing linguistic structure in machine learning models.


|    Method           |    Configuration                                                                         |    Accuracy     |    Precision    |    Recall    |    F1       |
|---------------------|------------------------------------------------------------------------------------------|-----------------|-----------------|--------------|-------------|
|    XLNET        |    large             |    0.916        |    0.872        |    0.973     |    0.920    |
|    COLBERT          |             |    0.982        |    0.990       |    0.974     |    0.982    |



## Trained model

If you do not want to train the model from scrach, you can download the following folder (my saved model) and put it in the same folder as your code. 

https://mega.nz/folder/MmB1gIIT#8ilUTK1-BO80aoXxKOIhpg

Then, you can use the following code to load the structure and weights of the model:

```py
import keras

model = keras.models.load_model("colbert-trained")
model.summary()
```

## Related paper

https://arxiv.org/abs/2004.12765
