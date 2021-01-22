[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/colbert-using-bert-sentence-embedding-for/humor-detection-on-200k-short-texts-for-humor-1)](https://paperswithcode.com/sota/humor-detection-on-200k-short-texts-for-humor-1?p=colbert-using-bert-sentence-embedding-for)

## Introduction

Automatic humor detection has interesting use cases in modern technologies, such as chatbots and personal assistants. In this paper, we describe a novel approach for detecting humor in short texts using BERT sentence embedding. Our proposed model uses BERT to generate sentence embeddings for texts which are sent as input to a neural network that predicts the target value. 

## Dependencies

- python 3.6
- transformers package

## Results

For evaluation, we created a new dataset for humor detection consisting of 200k formal short texts (100k positive, 100k negative). Experimental results show an accuracy of 98.1 percent for the proposed method, 2.1 percent improvement compared to the best CNN and RNN models and 1.1 percent better than a fine-tuned BERT model. In addition, the combination of RNN-CNN was not successful in this task compared to the CNN model.


|    Method           |    Configuration                                                                         |    Accuracy     |    Precision    |    Recall    |    F1       |
|---------------------|------------------------------------------------------------------------------------------|-----------------|-----------------|--------------|-------------|
|    XLNET        |    large             |    0.916        |    0.872        |    0.973     |    0.920    |
|    COLBERT          |             |    0.982        |    0.990       |    0.974     |    0.982    |


## Related paper: 

https://arxiv.org/abs/2004.12765
