import keras
from transformers import BertTokenizer
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.metrics import f1_score
from hri_tools import HumorDataset
from hri_tools import SUPPORTED_DATASETS
from tqdm import tqdm
import numpy as np
import pandas as pd

nltk.download('punkt')

COLBERT_PATH = "/home/ambaranov/Colbert/colbert-trained"
training_sample_count = 1000
test_count = 1000

MAX_SENTENCE_LENGTH = 20
MAX_SENTENCES = 5
MAX_LENGTH = 100
THRESHOLD = 0.5
MODEL_TYPE = 'bert-base-uncased'
input_categories = ["text"]


model = keras.models.load_model(COLBERT_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE)


def return_id(str1, str2, truncation_strategy, length):

    inputs = tokenizer.encode_plus(str1, str2,
        add_special_tokens=True,
        max_length=length,
        truncation_strategy=truncation_strategy)

    input_ids =  inputs["input_ids"]
    input_masks = [1] * len(input_ids)
    input_segments = inputs["token_type_ids"]
    padding_length = length - len(input_ids)
    padding_id = tokenizer.pad_token_id
    input_ids = input_ids + ([padding_id] * padding_length)
    input_masks = input_masks + ([0] * padding_length)
    input_segments = input_segments + ([0] * padding_length)

    return [input_ids, input_masks, input_segments]


def compute_input_arrays(df, columns, tokenizer):
    model_input = []
    for xx in range((MAX_SENTENCES*3)+3):
        model_input.append([])
    
    for _, row in df[columns].iterrows():
        i = 0
        
        sentences = sent_tokenize(row.text)
        for xx in range(MAX_SENTENCES):
            s = sentences[xx] if xx<len(sentences) else ''
            ids_q, masks_q, segments_q = return_id(s, None, 'longest_first', MAX_SENTENCE_LENGTH)
            model_input[i].append(ids_q)
            i+=1
            model_input[i].append(masks_q)
            i+=1
            model_input[i].append(segments_q)
            i+=1
        
        ids_q, masks_q, segments_q = return_id(row.text, None, 'longest_first', MAX_LENGTH)
        model_input[i].append(ids_q)
        i+=1
        model_input[i].append(masks_q)
        i+=1
        model_input[i].append(segments_q)
        
    for xx in range((MAX_SENTENCES*3)+3):
        model_input[xx] = np.asarray(model_input[xx], dtype=np.int32)
        
    print(model_input[0].shape)
    return model_input


result = list()

for dataset in tqdm(SUPPORTED_DATASETS):
    hri_dataset = HumorDataset(dataset)
    hri_dataset.load()
    test_inputs = compute_input_arrays(
        hri_dataset.get_test(),
        input_categories,
        tokenizer
    )
    test_preds = model.predict(test_inputs)
    preds_label = test_preds > THRESHOLD
    result.append(
        (dataset, f1_score(hri_dataset.get_test()["label"], preds_label))
    )

    df = pd.DataFrame(result, columns=['dataset', 'f1_score'])
    df.to_csv("hri_test_result.csv")
