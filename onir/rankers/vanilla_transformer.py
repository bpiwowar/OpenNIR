import torch
import torch.nn.functional as F
from onir import rankers, log
from experimaestro import param, config, Choices

@param("combine", default="linear", checker=Choices(["linear", "probe"]))
@param('outputs', default=1, help="Number of output for the classifier (for linear only)")
@config()
class VanillaTransformer(rankers.Ranker):
    """
    Implementation of the Vanilla BERT model from:
      > Sean MacAvaney, Andrew Yates, Arman Cohan, and Nazli Goharian. 2019. CEDR: Contextualized
      > Embeddings for Document Ranking. In SIGIR.
    Should be used with a transformer vocab, e.g., BertVocab.
    """
    def initialize(self, random):
        super().initialize(random)
        self.logger = log.easy()
        self.encoder = self.vocab.encoder()
        self.dropout = torch.nn.Dropout(0.1) # self.encoder.bert.config.hidden_dropout_prob
        if self.combine == 'linear':
            self.ranker = torch.nn.Linear(self.encoder.dim(), self.outputs)
        elif self.combine in ('prob', 'logprob'):
            assert self.outputs == 1
            self.ranker = torch.nn.Linear(self.encoder.dim(), 2)
        else:
            raise ValueError(f'unsupported combine={self.combine}')

    def __validate__(self):
        """Validate the parameters (called by experimaestro)"""
        assert self.vocab.__has_clstoken__, \
               "VanillaBert must be used with a vocab that supports CLS encoding (e.g., BertVocab)"
        # TODO: not really needed?
        # if self.vocab.encoder.static():
        #     logger.warn("It's usually bad to use VanillaBert with non-trainable embeddings. "
        #                 "Consider setting `vocab.train=True`")

    def input_spec(self):
        result = super().input_spec()
        result['fields'].update({'query_tok', 'query_len', 'doc_tok', 'doc_len'})
        result['qlen_mode'] = 'max'
        result['dlen_mode'] = 'max'
        return result

    def _forward(self, **inputs):
        pooled_output = self.encoder.enc_query_doc(**inputs)['cls'][-1]
        pooled_output = self.dropout(pooled_output)
        result = self.ranker(pooled_output)
        if self.combine == 'prob':
            result = result.softmax(dim=1)[:, 1]
        elif self.combine == 'logprob':
            result = result.log_softmax(dim=1)[:, 1]
        return result
