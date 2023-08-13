# -*- coding: utf-8 -*-

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel,AutoTokenizer
import os
from torch.utils import data
import numpy as np
import sys

max_seq_length = 512


class InputExample(object):

    def __init__(self, guid, words, labels):
        self.guid = guid
        self.words = words
        self.labels = labels

class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids,  predict_mask, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.predict_mask = predict_mask
        self.label_ids = label_ids
class DataProcessor(object):

    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()
    def get_predict_examples(self, data_dir,predict_string):
        raise NotImplementedError()
    def get_labels(self):
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file,isPredict = False,sentence = ''):
        if isPredict == False:
          with open(input_file) as f:
              out_lists = []
              entries = f.read().strip().split("\n\n")
              for entry in entries:
                  words = []
                  ner_labels = []
                  pos_tags = []
                  bio_pos_tags = []
                  for line in entry.splitlines():
                      pieces = line.strip().split()
                      if len(pieces) < 1:
                          continue
                      word = pieces[0]
                      words.append(word)
                      ner_labels.append(pieces[-1])
                  
                  out_lists.append([words,pos_tags,bio_pos_tags,ner_labels])
        else:
          out_lists = []
          words = []
          ner_labels = []
          pos_tags = []
          bio_pos_tags = []
          entries = sentence.strip().split(" ")
          for i in entries:
            if len(i) < 1:
                continue
            word = i
            words.append(word)
            ner_labels.append('O')
          out_lists.append([words,pos_tags,bio_pos_tags,ner_labels])
        return out_lists

class DNRTIDataProcessor(DataProcessor):

    def __init__(self):
        self._label_types =  [ 'X', '[CLS]', '[SEP]', 'O', 'B-Area', 'B-Exp', 'B-Features', 'B-HackOrg', 'B-Idus', 'B-OffAct','B-Org', 'B-Purp', 'B-SamFile','B-SecTeam','B-Time','B-Tool','B-Way','I-Area','I-Exp','I-Features','I-HackOrg','I-Idus','I-OffAct','I-Org','I-Purp','I-SamFile','I-SecTeam','I-Time','I-Tool','I-Way']
        self._num_labels = len(self._label_types)
        self._label_map = {label: i for i,
                           label in enumerate(self._label_types)}

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "train.txt")))

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "valid.txt")))
    def get_predict_examples(self,data_dir, predict_string):
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "None.txt"),True,predict_string),True)
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "test.txt")))

    def get_labels(self):
        return self._label_types

    def get_num_labels(self):
        return self.get_num_labels

    def get_label_map(self):
        return self._label_map

    def get_start_label_id(self):
        return self._label_map['[CLS]']

    def get_stop_label_id(self):
        return self._label_map['[SEP]']

    def _create_examples(self, all_lists,isPredict = False):
        examples = []
        if isPredict == False:
          for (i, one_lists) in enumerate(all_lists):
            
            guid = i
            words = one_lists[0]
            labels = one_lists[-1]

            examples.append(InputExample(
                guid=guid, words=words, labels=labels))
            
        else:
          k = 1
          for i in all_lists:

            guid = k
            k += 1
            words = i[0]
            labels = i[3]
            examples.append(InputExample(
                guid=guid, words=words, labels=labels))

        return examples

    def _create_examples2(self, lines):
        examples = []
        for (i, line) in enumerate(lines):
            guid = i
            text = line[0]
            ner_label = line[-1]
            examples.append(InputExample(
                guid=guid, text_a=text, labels_a=ner_label))
        return examples
def example2feature(example, tokenizer, label_map, max_seq_length):
    add_label = 'X'
    # tokenize_count = []
    tokens = ['[CLS]']
    predict_mask = [0]
    label_ids = [label_map['[CLS]']]
    for i, w in enumerate(example.words):
        sub_words = tokenizer.tokenize(w)
        if not sub_words:
            sub_words = ['[UNK]']
        tokens.extend(sub_words)
        for j in range(len(sub_words)):
            if j == 0:
                predict_mask.append(1)
                label_ids.append(label_map[example.labels[i]])
            else:
                predict_mask.append(0)
                label_ids.append(label_map[add_label])

    if len(tokens) > max_seq_length - 1:
        print('Example No.{} is too long, length is {}, truncated to {}!'.format(example.guid, len(tokens), max_seq_length))
        tokens = tokens[0:(max_seq_length - 1)]
        predict_mask = predict_mask[0:(max_seq_length - 1)]
        label_ids = label_ids[0:(max_seq_length - 1)]
    tokens.append('[SEP]')
    predict_mask.append(0)
    label_ids.append(label_map['[SEP]'])

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)
    feat=InputFeatures(
                # guid=example.guid,
                # tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                predict_mask=predict_mask,
                label_ids=label_ids)

    return feat
class NerDataset(data.Dataset):
    def __init__(self, examples, tokenizer, label_map, max_seq_length):
        self.examples=examples
        self.tokenizer=tokenizer
        self.label_map=label_map
        self.max_seq_length=max_seq_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        feat=example2feature(self.examples[idx], self.tokenizer, self.label_map, max_seq_length)
        return feat.input_ids, feat.input_mask, feat.segment_ids, feat.predict_mask, feat.label_ids

    @classmethod
    def pad(cls, batch):

        seqlen_list = [len(sample[0]) for sample in batch]
        maxlen = np.array(seqlen_list).max()

        f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]
        input_ids_list = torch.LongTensor(f(0, maxlen))
        input_mask_list = torch.LongTensor(f(1, maxlen))
        segment_ids_list = torch.LongTensor(f(2, maxlen))
        predict_mask_list = torch.ByteTensor(f(3, maxlen))
        label_ids_list = torch.LongTensor(f(4, maxlen))

        return input_ids_list, input_mask_list, segment_ids_list, predict_mask_list, label_ids_list

def f1_score(y_true, y_pred):
    ignore_id=3

    num_proposed = len(y_pred[y_pred>ignore_id])
    num_correct = (np.logical_and(y_true==y_pred, y_true>ignore_id)).sum()
    num_gold = len(y_true[y_true>ignore_id])

    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        if precision*recall==0:
            f1=1.0
        else:
            f1=0

    return precision, recall, f1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BertLayerNorm = torch.nn.LayerNorm
def log_sum_exp_1vec(vec):  
    max_score = vec[0, np.argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def log_sum_exp_mat(log_M, axis=-1): 
    return torch.max(log_M, axis)[0]+torch.log(torch.exp(log_M-torch.max(log_M, axis)[0][:, None]).sum(axis))

def log_sum_exp_batch(log_Tensor, axis=-1): 
    return torch.max(log_Tensor, axis)[0]+torch.log(torch.exp(log_Tensor-torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0],-1,1)).sum(axis))

class BERT_CRF_NER(nn.Module):

    def __init__(self, bert_model, start_label_id, stop_label_id, num_labels, max_seq_length, batch_size, device):
        super(BERT_CRF_NER, self).__init__()
        self.hidden_size = 768
        self.start_label_id = start_label_id
        self.stop_label_id = stop_label_id
        self.num_labels = num_labels
        # self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.device=device
        self.bert = bert_model
        self.dropout = torch.nn.Dropout(0.2)
        self.hidden2label = nn.Linear(self.hidden_size, self.num_labels)
        self.transitions = nn.Parameter(torch.randn(self.num_labels, self.num_labels))
        self.transitions.data[start_label_id, :] = -10000
        self.transitions.data[:, stop_label_id] = -10000

        nn.init.xavier_uniform_(self.hidden2label.weight)
        nn.init.constant_(self.hidden2label.bias, 0.0)


    def init_bert_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _forward_alg(self, feats):
        T = feats.shape[1]
        batch_size = feats.shape[0]
        log_alpha = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(self.device)
        log_alpha[:, 0, self.start_label_id] = 0
        for t in range(1, T):
            log_alpha = (log_sum_exp_batch(self.transitions + log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)
        log_prob_all_barX = log_sum_exp_batch(log_alpha)
        return log_prob_all_barX

    def _get_bert_features(self, input_ids, segment_ids, input_mask):
        bert_seq_out, _ = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,return_dict=False)
        bert_seq_out = self.dropout(bert_seq_out)
        bert_feats = self.hidden2label(bert_seq_out)
        return bert_feats

    def _score_sentence(self, feats, label_ids):
        T = feats.shape[1]
        batch_size = feats.shape[0]

        batch_transitions = self.transitions.expand(batch_size,self.num_labels,self.num_labels)
        batch_transitions = batch_transitions.flatten(1)

        score = torch.zeros((feats.shape[0],1)).to(device)
        for t in range(1, T):
            score = score + \
                batch_transitions.gather(-1, (label_ids[:, t]*self.num_labels+label_ids[:, t-1]).view(-1,1)) \
                    + feats[:, t].gather(-1, label_ids[:, t].view(-1,1)).view(-1,1)
        return score

    def _viterbi_decode(self, feats):
        T = feats.shape[1]
        batch_size = feats.shape[0]
        log_delta = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(self.device)
        log_delta[:, 0, self.start_label_id] = 0
        psi = torch.zeros((batch_size, T, self.num_labels), dtype=torch.long).to(self.device)  
        for t in range(1, T):
            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)
        path = torch.zeros((batch_size, T), dtype=torch.long).to(self.device)
        max_logLL_allz_allx, path[:, -1] = torch.max(log_delta.squeeze(), -1)

        for t in range(T-2, -1, -1):
            path[:, t] = psi[:, t+1].gather(-1,path[:, t+1].view(-1,1)).squeeze()
        return max_logLL_allz_allx, path

    def neg_log_likelihood(self, input_ids, segment_ids, input_mask, label_ids):
        bert_feats = self._get_bert_features(input_ids, segment_ids, input_mask)
        forward_score = self._forward_alg(bert_feats)
        gold_score = self._score_sentence(bert_feats, label_ids)
        return torch.mean(forward_score - gold_score)

    def forward(self, input_ids, segment_ids, input_mask):
        bert_feats = self._get_bert_features(input_ids, segment_ids, input_mask)
        score, label_seq_ids = self._viterbi_decode(bert_feats)
        return score, label_seq_ids



bert_model_scale = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(bert_model_scale, do_lower_case=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DNRTIProcessor = DNRTIDataProcessor()
start_label_id = DNRTIProcessor.get_start_label_id()
stop_label_id = DNRTIProcessor.get_stop_label_id()
label_list = DNRTIProcessor.get_labels()
label_map = DNRTIProcessor.get_label_map()
batch_size = 512
bert_model = BertModel.from_pretrained(bert_model_scale)
model = BERT_CRF_NER(bert_model, start_label_id, stop_label_id, len(label_list), max_seq_length, batch_size, device)


checkpoint = torch.load('./outputs/ner_bert_crf_checkpoint.pt', map_location='cpu')
epoch = checkpoint['epoch']
valid_acc_prev = checkpoint['valid_acc']
valid_f1_prev = checkpoint['valid_f1']
pretrained_dict=checkpoint['model_state']
net_state_dict = model.state_dict()
pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
net_state_dict.update(pretrained_dict_selected)
model.load_state_dict(net_state_dict)
print('Loaded the pretrain  NER_BERT_CRF  model, epoch:',checkpoint['epoch'],'valid acc:',
      checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])

model.to(device)


max_seq_length = 512
data_dir = 'none.txt'

def predict_sentence(sentence):
    predict_string = sentence
    predict_examples = DNRTIProcessor.get_predict_examples(data_dir,predict_string = predict_string)
    predict_dataset = NerDataset(predict_examples,tokenizer,label_map,max_seq_length)

    # model.eval()
    with torch.no_grad():
        demon_dataloader = data.DataLoader(dataset=predict_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=1,
                                    collate_fn=NerDataset.pad)
        for batch in demon_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
            _, predicted_label_seq_ids = model(input_ids, segment_ids, input_mask)
            # _, predicted = torch.max(out_scores, -1)
            valid_predicted = torch.masked_select(predicted_label_seq_ids, predict_mask)
            # valid_label_ids = torch.masked_select(label_ids, predict_mask)
            for i in range(1):
                new_ids=predicted_label_seq_ids[i].cpu().numpy()[predict_mask[i].cpu().numpy()==1]
                # print(list(map(lambda i: label_list[i], new_ids)))
                xx = list(map(lambda i: label_list[i], new_ids))
                result = " ".join(str(x) for x in list(map(lambda i: label_list[i], new_ids)))
                for sss in xx:
                    if sss != "O":
                        pass
                print("----------------")
                print("source sentence: ", running_input)
                print("result sentence: ", result)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-I', '--input', required=True, type=str, default="", help="Please enter the OSINT or CTI senetences.")
    arguments = parser.parse_args(sys.argv[1:])
    running_input = arguments.input
    if running_input:
        predict_sentence(running_input)
    else:
        print("Error!")