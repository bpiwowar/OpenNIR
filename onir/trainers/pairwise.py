import sys
import torch
import torch.nn.functional as F
from experimaestro import argument, config, Choices
import onir
from onir import trainers, spec, util

@argument(
    "lossfn", type=str, default="softmax", checker=Choices(['softmax', 'cross_entropy', 'hinge'])
)  
@argument("pos_source", type=str, default="intersect", checker=Choices(['intersect', 'qrels']))
@argument("neg_source", type=str, default="neg_source", checker=Choices(['run', 'qrels', 'union']))
@argument("sampling", type=str, default="query", checker=Choices(['query', 'qrel']))
@argument("pos_minrel", default=1)
@argument("unjudged_rel", default=0)
@argument("num_neg", default=1)
@argument("margin", default=0.0)
@config()
class PairwiseTrainer(trainers.Trainer):
    def __init__(self, logger, train_ds, vocab, random):
        super().__init__(config, ranker, vocab, train_ds, logger, random)
        self.loss_fn = {
            'softmax': self.softmax,
            'cross_entropy': self.cross_entropy,
            'nogueira_cross_entropy': self.nogueira_cross_entropy,
            'hinge': self.hinge
        }[self.lossfn]
        self.dataset = train_ds
        self.input_spec = ranker.input_spec()
        self.iter_fields = self.input_spec['fields'] | {'runscore'}
        self.train_iter_core = onir.datasets.pair_iter(
            train_ds,
            fields=self.iter_fields,
            pos_source=self.pos_source,
            neg_source=self.neg_source,
            sampling=self.sampling,
            pos_minrel=self.pos_minrel,
            unjudged_rel=self.unjudged_rel,
            num_neg=self.num_neg,
            random=self.random,
            inf=True)
        self.train_iter = util.background(self.iter_batches(self.train_iter_core))
        self.numneg = config['num_neg']

    def path_segment(self):
        path = super().path_segment()
        pos = 'pos-{pos_source}-{sampling}'.format(**self.config)
        if self.pos_minrel != 1:
            pos += '-minrel{pos_minrel}'.format(**self.config)
        neg = 'neg-{neg_source}'.format(**self.config)
        if self.unjudged_rel != 0:
            neg += '-unjudged{unjudged_rel}'.format(**self.config)
        if self.num_neg != 1:
            neg += '-numneg{num_neg}'.format(**self.config)
        loss = self.lossfn
        if loss == 'hinge':
            loss += '-{margin}'.format(**self.config)
        result = 'pairwise_{path}_{loss}_{pos}_{neg}'.format(**self.config, loss=loss, pos=pos, neg=neg, path=path)
        if self.gpu and not self.gpu_determ:
            result += '_nondet'
        return result

    def iter_batches(self, it):
        while True: # breaks on StopIteration
            input_data = {}
            for _, record in zip(range(self.batch_size), it):
                for k, v in record.items():
                    assert len(v) == self.numneg + 1
                    for seq in v:
                        input_data.setdefault(k, []).append(seq)
            input_data = spec.apply_spec_batch(input_data, self.input_spec, self.device)
            yield input_data

    def train_batch(self):
        input_data = next(self.train_iter)
        rel_scores = self.ranker(**input_data)
        if torch.isnan(rel_scores).any() or torch.isinf(rel_scores).any():
            self.logger.error('nan or inf relevance score detected. Aborting.')
            sys.exit(1)
        rel_scores_by_record = rel_scores.reshape(self.batch_size, self.numneg + 1, -1)
        run_scores_by_record = input_data['runscore'].reshape(self.batch_size, self.numneg + 1)
        loss = self.loss_fn(rel_scores_by_record)
        losses = {'data': loss}
        loss_weights = {'data': 1.}

        return {
            'losses': losses,
            'loss_weights': loss_weights,
            'acc': self.acc(rel_scores_by_record),
            'unsup_acc': self.acc(run_scores_by_record)
        }

    def fast_forward(self, record_count):
        self._fast_forward(self.train_iter_core, self.iter_fields, record_count)

    @staticmethod
    def cross_entropy(rel_scores_by_record):
        target = torch.zeros(rel_scores_by_record.shape[0]).long().to(rel_scores_by_record.device)
        return F.cross_entropy(rel_scores_by_record, target, reduction='mean')

    @staticmethod
    def nogueira_cross_entropy(rel_scores_by_record):
        """
        cross entropy loss formulation for BERT from:
         > Rodrigo Nogueira and Kyunghyun Cho. 2019.Passage re-ranking with bert. ArXiv,
         > abs/1901.04085.
        """
        log_probs = -rel_scores_by_record.log_softmax(dim=2)
        return (log_probs[:, 0, 0] + log_probs[:, 1, 1]).mean()

    @staticmethod
    def softmax(rel_scores_by_record):
        return torch.mean(1. - F.softmax(rel_scores_by_record, dim=1)[:, 0])

    def hinge(self, rel_scores_by_record):
        return F.relu(self.margin - rel_scores_by_record[:, :1] + rel_scores_by_record[:, 1:]).mean()

    @staticmethod
    def pointwise(rel_scores_by_record):
        log_probs = -rel_scores_by_record.log_softmax(dim=2)
        return (log_probs[:, 0, 0] + log_probs[:, 1, 1]).mean()

    @staticmethod
    def acc(scores_by_record):
        count = scores_by_record.shape[0] * (scores_by_record.shape[1] - 1)
        return (scores_by_record[:, :1] > scores_by_record[:, 1:]).sum().float() / count
