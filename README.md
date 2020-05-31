[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/colbert-using-bert-sentence-embedding-for/humor-detection-on-200k-short-texts-for-humor-1)](https://paperswithcode.com/sota/humor-detection-on-200k-short-texts-for-humor-1?p=colbert-using-bert-sentence-embedding-for)

## Introduction

Automatic humor detection has interesting use cases in modern technologies, such as chatbots and personal assistants. In this paper, we describe a novel approach for detecting humor in short texts using BERT sentence embedding. Our proposed model uses BERT to generate tokens and sentence embedding for texts. It sends embedding outputs as input to a two-layered neural network that predicts the target value. 

## Dependencies

- python 3.6
- transformers package

## Results

For evaluation, we created a new dataset for humor detection consisting of 200k formal short texts (100k positive, 100k negative). Experimental results show an accuracy of 98.1 percent for the proposed method, 2.1 percent improvement compared to the best CNN and RNN models and 1.1 percent better than a fine-tuned BERT model. In addition, the combination of RNN-CNN was not successful in this task compared to the CNN model.


|    Method           |    Configuration                                                                         |    Accuracy     |    Precision    |    Recall    |    F1       |
|---------------------|------------------------------------------------------------------------------------------|-----------------|-----------------|--------------|-------------|
|    CNN              |    cnn_drop_out = 0.2                                                                    |    0.960        |    0.955        |    0.966     |    0.960    |
|    Attention RNN    |    rnn_depth=1   rnn_drop_out = 0.3   rnn_state_drop_out = 0.3                           |    0.958        |    0.977        |    0.939     |    0.957    |
|    BiRNN - CNN      |    rnn_depth=1   cnn_drop_out = 0.2   rnn_drop_out = 0.3   rnn_state_drop_out =   0.3    |    0.960        |    0.954        |    0.965     |    0.960    |
|    BERT-base        |    uncased   LR=1.5e-4                                                                   |    0.969        |    0.965        |    0.975     |    0.970    |
|    COLBERT          |    BERT-based-uncased   LR=1.5e-4   NN-L1-dropout=0.2   NN-L1-activation=sigmoid         |    0.981        |    0.984        |    0.977     |    0.981    |


## Dataset: 

https://www.kaggle.com/moradnejad/200k-short-texts-for-humor-detection

## Related paper: 

https://arxiv.org/abs/2004.12765
