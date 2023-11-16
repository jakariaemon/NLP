import torch
import torch.nn.functional as F

import onnx
import onnxruntime
import numpy as np
from transformers import (WEIGHTS_NAME, BertConfig,
                          BertForSequenceClassification, BertTokenizer)
from sklearn.metrics import accuracy_score
import time
import pandas as pd

"""
This script converts pretrained pytorch bert model to onnx
and compares the performance between pytorch and onnx model
"""



def preprocess(tokenizer, sent):
    tokenizer = BertTokenizer.from_pretrained("/Users/jakar/Downloads/model/mm")
    encoded_dict = tokenizer.encode_plus(
        sent,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=64,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )
    input_id = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    input_id = torch.LongTensor(input_id)
    attention_mask = torch.LongTensor(attention_mask)

    return input_id, attention_mask 

"""
Load the test dataset
"""

def load_data(file_):

    read_file = open(file_)
    examples =[] 
    labels =[]
    df = pd.read_csv(file_, delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    print('Number of training sentences: {:,}\n'.format(df.shape[0]))
    examples = df.sentence.values
    labels = df.label.values


    return examples, labels

"""
Inference on pretrained pytorch model
"""

def inference_pytorch(model_loaded, input_id, attention_mask):

    with torch.no_grad():
        outputs = model_loaded(input_id, attention_mask)

    logits = outputs[0]
    index = logits.argmax()
    return index

"""
This function stores pretrained bert model
into onnx format
"""

def convert_bert_to_onnx(sent):

    model_dir = "/Users/jakar/Downloads/model/mm"
    config = BertConfig.from_pretrained(model_dir)
    model_loaded = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir, config=config)
    model.to("cpu")
    input_id, attention_mask = preprocess(tokenizer, sent)

    torch.onnx.export(model, (input_id, attention_mask), "bert.onnx",  input_names = ["input_id", "attention_mask"],
    output_names = ["output"])

    print("model convert to onnx format successfully")


def inference(model_name, examples):

    onnx_inference = []
    pytorch_inference = []
    model_dir = "/Users/jakar/Downloads/model/mm"
    #onnx session
    ort_session = onnxruntime.InferenceSession(model_name)
    #pytorch pretrained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    config = BertConfig.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir, config=config)
    model.to("cpu")

    for example in examples:
        """
        Onnx inference
        """
        input_id, attention_mask = preprocess(tokenizer, examples)
        #
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_ids),
                        ort_session.get_inputs()[1].name: to_numpy(attention_mask)}
        ort_outs = ort_session.run(["output"], ort_inputs)
        torch_onnx_output = torch.tensor(ort_outs[0], dtype=torch.float32)
        onnx_logits = F.softmax(torch_onnx_output, dim=1)

        logits_label = torch.argmax(onnx_logits, dim=1)
        label = logits_label.detach().cpu().numpy()
        onnx_inference.append(label[0])

        """
        Pretrained bert pytorch model
        """
        #

        torch_out = inference_pytorch(model, input_id, attention_mask)

        logits_label = torch.argmax(torch_out, dim=1)
        label = logits_label.detach().cpu().numpy()
        pytorch_inference.append(label[0])


        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(to_numpy(torch_out), onnx_logits, rtol=1e-03, atol=1e-05)

        print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    return onnx_inference, pytorch_inference


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

if __name__ == '__main__':

    sent = "i is a student"
    convert_bert_to_onnx(sent)

    examples, labels = load_data("/Users/jakar/Downloads/model/dev.tsv")
    # start_time = time.time()
    # print("labels ", labels)

    # returns results from pytorch pretrained model and onnx
    onnx_labels, pytorch_labels = inference("bert.onnx", examples)
    print("\n ************ \n")

    # print("total time ", time.time() - start_time)
    print("accuracy score of onnx model", accuracy_score(labels, onnx_labels))
    print("accuracy score of pytorch model", accuracy_score(labels, pytorch_labels))
