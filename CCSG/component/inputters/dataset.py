# src: https://github.com/facebookresearch/DrQA/blob/master/drqa/reader/data.py

import random

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from component.inputters.vector import vectorize
import json
from component.inputters import utils
from .constants import edge_type
from .constants import EOS,BOS
import torch
from component.inputters.vocabulary import Vocabulary

BUFSIZE = 409600000  # 400MB
class DataLoader(object):
    def __init__(self, filename, batch_size, max_len):
        self.batch_size = batch_size
        self.max_len = max_len
        self.filename = filename
        self.stream = open(self.filename, encoding='utf8')
        self.epoch_id = 0

    def __iter__(self):
        lines = self.stream.readlines(BUFSIZE)
        if not lines:
            self.epoch_id += 1
            self.stream.close()
            self.stream = open(self.filename, encoding='utf8')
            lines = self.stream.readlines(BUFSIZE)

        # for json
        docs = []
        for line in lines:
            line = eval(line)
            line = line['content']
            line = [self.tokenizer.tokenize(l.strip()) for l in line.split('\n') if l.strip()]
            docs.append(line)
        docs = [x for x in docs if len(x) > 2]  # 筛掉一些短文章，很关键
        random.shuffle(docs)
        # end for json

        data = []
        for idx, doc in enumerate(docs):
            data.extend(self.create_instances_from_document(docs, idx, self.max_len))

        idx = 0
        while idx < len(data):
            yield self.convert_to_features(data[idx:idx + self.batch_size], self.tokenizer, self.encode_type,
                                           self.max_len)
            idx += self.batch_size
# ------------------------------------------------------------------------------
# PyTorch dataset class for SQuAD (and SQuAD-like) data.
# ------------------------------------------------------------------------------
class InCompleteCodeDataset(Dataset):
    def __init__(self, examples,args,vocabs):
        self.examples = examples['reses']
        self.args=args
        (self.vocab_attr, self.vocab_type, self.vocab_token)=vocabs

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        item=self.convert_vectors(self.examples[index])
        return item

    def lengths(self):
        return len(self.examples)

    def convert_vectors(self,ex):


        node_len=len(ex["tokens"])+len(ex["types"])
        edge_dicts={}
        for key in edge_type.keys():
            if key in ex['edges'].keys():
                es=ex['edges'][key]
            else:es=[]
            edge_metrix=torch.zeros([node_len,node_len])
            if len(es)>0:
                index_edges=torch.tensor([e for e in es],dtype=torch.long).T
                edge_metrix=edge_metrix.index_put((index_edges[0],index_edges[1]),torch.ones(index_edges.shape[1]))
            edge_dicts[key] = edge_metrix

        src_vocab=Vocabulary(no_special_token=True)
        src_vocab.add_tokens([str(i) for i in ex['text'][0]])
        src_map=torch.tensor([src_vocab[str(i)] for i in ex['text'][0]],dtype=torch.int64)
        alignments=torch.tensor([BOS]+[src_vocab[str(i.value)] for i in ex['text'][1]] +[EOS],dtype=torch.int64)


        node_tokens = torch.tensor([self.vocab_token[str(i[0])] for i in ex['tokens']],
                                dtype=torch.int64)

        node_types = torch.tensor([self.vocab_type[i[0]] for i in ex['types']],
                                dtype=torch.int64)

        attrs=[i[1] for i in ex['types']] + [i[1] for i in ex['tokens']]
        node_attr = torch.tensor([self.vocab_attr[i] for i in attrs],
                                dtype=torch.int64)

        text_src=torch.tensor([self.vocab_token[i] for i in ex["text"][0]],
                                dtype=torch.int64)
        text_tgt=torch.tensor([BOS]+[self.vocab_token[str(i.value)] for i in ex["text"][1]]+[EOS],
                                dtype=torch.int64)


        item={
            "tokens":node_tokens,
            "types":node_types,
            "edge_dicts":edge_dicts,
            "MASK_id":ex['MASK_id'],
            "text_src":text_src,
            "text_tgt":text_tgt,
            "attrs":node_attr,
            "src_vocab":src_vocab,
            "src_map":src_map,
            "alignments":alignments,
            "raw_text":ex["text"][0],
            "raw_tgt":ex["text"][1]
        }
        return item





class CombineDataset(Dataset):
    def __init__(self, examples, model,args,eval_types=None):
        self.model = model
        self.examples = examples[0]
        self.args=args
        self.eval_types=eval_types

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        json_object=json.loads(self.examples[index])
        args=self.args
        example=utils.process_examples(0,
                                   json_object['target'].replace('@@ ',""),
                                   json_object,
                                   args.max_src_len,
                                   args.max_tgt_len,
                                   args.code_tag_type,
                                   uncase=args.uncase,
                                   test_split=False,
                                   tgt_bpe=json_object['target'] if args.use_bpe else None, MTL=args.MTL)
        vectorized_ex = vectorize(example, self.model)
        vectorized_ex['eval_type']=self.eval_types[index] if self.eval_types is not None else None
        return vectorized_ex

    def lengths(self):
        return len(self.examples)

class CommentDataset(Dataset):
    def __init__(self, examples, model):
        self.model = model
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return vectorize(self.examples[index], self.model)

    def lengths(self):
        return [(len(ex['gnn'].code_tokens), len(ex['targetCode'].tokens))
                for ex in self.examples]


# ------------------------------------------------------------------------------
# PyTorch sampler returning batched of sorted lengths (by doc and question).
# ------------------------------------------------------------------------------


class SortedBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        lengths = np.array(
            [(-l[0], -l[1], np.random.random()) for l in self.lengths],
            dtype=[('l1', np.int_), ('l2', np.int_), ('rand', np.float_)]
        )
        indices = np.argsort(lengths, order=('l1', 'l2', 'rand'))
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return len(self.lengths)
