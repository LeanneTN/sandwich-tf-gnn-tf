"""CCSG model"""
import copy
import math
import logging
from tqdm import tqdm

import torch
import torch.optim as optim

import component.config
from torch.nn.utils import clip_grad_norm_

import config
from component.config import override_model_args
from component.models.transformer import Transformer
from component.utils.copy_utils import collapse_copy_scores, replace_unknown, \
    make_src_map, align
from component.utils.misc import tens2sen, count_file_lines

logger = logging.getLogger(__name__)


class CCSGModel(object):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self, args, vocabs,state_dict=None):
        # Book-keeping.
        self.args = args
        self.args.src_vocab_size = 0
        (self.vocab_attr, self.vocab_type, self.vocab_token)=vocabs
        self.args.vocab_size = len(self.vocab_token)
        self.args.type_vocab_size=len(self.vocab_type)
        self.args.attr_vocab_size = len(self.vocab_attr)
        self.updates = 0
        self.use_cuda = False
        self.parallel = False
        self.network = Transformer(args, self.vocab_token)


        # Load saved state
        if state_dict is not None:
            # Load buffer separately
            self.network.load_state_dict(state_dict)

    def init_optimizer(self, state_dict=None, use_gpu=True):
        """Initialize an optimizer for the free parameters of the network.
        Args:
            state_dict: optimizer's state dict
            use_gpu: required to move state_dict to GPU
        """
        if self.args.fix_embeddings:
            self.network.embedder.src_word_embeddings.fix_word_lut()
            self.network.embedder.tgt_word_embeddings.fix_word_lut()

        if self.args.optimizer == 'sgd':
            parameters = [p for p in self.network.parameters() if p.requires_grad]
            self.optimizer = optim.SGD(parameters,
                                       self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)

        elif self.args.optimizer == 'adam':
            parameters = [p for p in self.network.parameters() if p.requires_grad]
            self.optimizer = optim.Adam(parameters,
                                        self.args.learning_rate,
                                        weight_decay=self.args.weight_decay)

        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.args.optimizer)

        if state_dict is not None:
            self.optimizer.load_state_dict(state_dict)
            # FIXME: temp soln - https://github.com/pytorch/pytorch/issues/2830
            if use_gpu:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

    # --------------------------------------------------------------------------
    # Learning
    # --------------------------------------------------------------------------

    def convert_cuda(self,data):
        if isinstance(data,tuple):
            lst=[]
            for i in data:
                lst.append(self.convert_cuda(i))
            return tuple(lst)
        if isinstance(data,list):
            for idx,i in enumerate(data):
                data[idx]=self.convert_cuda(i)
            return data
        elif isinstance(data,dict):
            for key,v in data.items():
                data[key]=self.convert_cuda(v)
            return data
        elif isinstance(data,torch.Tensor):
            return data.cuda(non_blocking=True)
        else:
            return data

    def update(self, ex, epoch):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

        # Train mode
        self.network.train()
        bsz=ex['batch_size']
        data=ex['data']
        (vec_type, vec_token, vec_src, vec_tgt, vec_attrs, vec_MASK), adjacency_m, (
            lengths_type, lengths_token, lengths_src, lengths_tgt, lengths_node
        ), (src_vocabs, src_map, alignments)=data


        source_map, alignment = None, None
        blank, fill = None, None
        # To enable copy attn, collect source map and alignment info
        if self.args.copy_attn:

            source_map = make_src_map(src_map)

            alignment = align(alignments)
            assert (alignments==alignment).all()

            blank, fill = collapse_copy_scores(self.vocab_token, src_vocabs)

        data_tuples=(data,(source_map,blank, fill))
        if self.args.use_cuda and not self.args.parallel:
            data_tuples=self.convert_cuda(data_tuples)


        # Run forward
        net_loss,attns = self.network(data_tuples)
        # GNN point
        # loss = net_loss['ml_loss'].mean() if self.parallel \
        #     else net_loss['ml_loss']
        # loss_per_token = net_loss['loss_per_token'].mean() if self.parallel \
        #     else net_loss['loss_per_token']

        loss = net_loss['ml_loss'].mean()
        loss_per_token = net_loss['loss_per_token'].mean()

        if loss.device.type!="cpu":
            ml_loss = loss.item()
            loss_per_token = loss_per_token.item()
        else:
            ml_loss = loss
        loss_per_token = 10 if loss_per_token > 10 else loss_per_token
        perplexity = math.exp(loss_per_token)

        loss.backward()

        clip_grad_norm_(self.network.parameters(), self.args.grad_clipping)
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.updates += 1
        return {
            'ml_loss': ml_loss,
            'perplexity': perplexity,
        }

    # --------------------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------------------

    def predict(self, ex, replace_unk=False):
        """Forward a batch of examples; step the optimizer to update weights."""

        # Train mode
        self.network.eval()
        bsz=ex['batch_size']
        raw_text=ex['raw_text']
        raw_tgt=ex['raw_tgt']
        data=ex['data']
        (vec_type, vec_token, vec_src, vec_tgt, vec_attrs, vec_MASK), adjacency_m, (
            lengths_type, lengths_token, lengths_src, lengths_tgt, lengths_node
        ), (src_vocabs, src_map, alignments)=data


        source_map, alignment = None, None
        blank, fill = None, None
        # To enable copy attn, collect source map and alignment info
        if self.args.copy_attn:

            source_map = make_src_map(src_map)

            alignment = align(alignments)
            assert (alignments==alignment).all()

            blank, fill = collapse_copy_scores(self.vocab_token, src_vocabs)

        data_tuples=(data,(source_map,blank, fill))
        if self.args.use_cuda and not self.args.parallel:
            data_tuples=self.convert_cuda(data_tuples)


        # Run forward
        decoder_out  = self.network(data_tuples)

        predictions = tens2sen(decoder_out['predictions'],
                               self.vocab_token,
                               src_vocabs)
        if replace_unk:
            for i in range(len(predictions)):
                enc_dec_attn = decoder_out['attentions'][i]
                assert enc_dec_attn.dim() == 3
                enc_dec_attn = enc_dec_attn.mean(1)
                predictions[i] = replace_unknown(predictions[i],
                                                 enc_dec_attn,
                                                 src_raw=raw_text[i])


        targets = [tarCode for tarCode in raw_tgt]
        return predictions, targets, decoder_out['copy_info']


    # --------------------------------------------------------------------------
    # Saving and loading
    # --------------------------------------------------------------------------

    def save(self, filename):
        """Save the current checkpoint
        """
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        state_dict = copy.copy(network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {
            'state_dict': state_dict,
            'vocab_attr': self.vocab_attr,
            'vocab_token': self.vocab_token,
            'vocab_type': self.vocab_type,
            'args': self.args,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    def checkpoint(self, filename, epoch):
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        params = {
            'state_dict': network.state_dict(),
            'vocab_type': self.vocab_type,
            'vocab_token': self.vocab_token,
            'vocab_attr': self.vocab_attr,
            'args': self.args,
            'epoch': epoch,
            'updates': self.updates,
            'optimizer': self.optimizer.state_dict(),
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load(filename, new_args=None):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        vocab_attr = saved_params['vocab_attr']
        vocab_token = saved_params['vocab_token']
        vocab_type = saved_params['vocab_type']
        state_dict = saved_params['state_dict']
        args = saved_params['args']
        if new_args:
            args = override_model_args(args, new_args)
        vocabs=(vocab_attr, vocab_type, vocab_token)
        return CCSGModel(args, vocabs,state_dict)

    @staticmethod
    def load_checkpoint(filename, use_gpu=True, original_args=None):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )

        vocab_type=saved_params['vocab_type']
        vocab_token = saved_params['vocab_token']
        vocab_attr = saved_params['vocab_attr']


        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch']
        updates = saved_params['updates']
        optimizer = saved_params['optimizer']
        args = saved_params['args']

        args_attr=set.union(config.MODEL_ARCHITECTURE,config.DATA_OPTIONS,config.MODEL_OPTIMIZER)
        for attr in args_attr:
            if not hasattr(args,attr):
                value=getattr(original_args,attr,False)
                setattr(args,attr,value)

        args.use_cuda=original_args.use_cuda
        args.parallel=original_args.parallel

        vocabs=(vocab_attr, vocab_type, vocab_token)
        model = CCSGModel(args, vocabs,state_dict)
        model.updates = updates
        model.init_optimizer(optimizer, use_gpu)
        return model, epoch

    # --------------------------------------------------------------------------
    # Runtime
    # --------------------------------------------------------------------------

    def cuda(self):
        self.use_cuda = True
        self.network = self.network.cuda()

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()

    def parallelize(self):
        """Use data parallel to copy the model across several gpus.
        This will take all gpus visible with CUDA_VISIBLE_DEVICES.
        """
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)
