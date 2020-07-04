import os
import json
import torch
from pathlib import Path
from typing import List
from experimaestro import config, param, pathoption
import onir
from onir import util, spec, predictors, datasets
from onir.interfaces import trec, plaintext
from onir.log import Logger
from onir.rankers.trivial import Trivial
from onir.util import Device, DEFAULT_DEVICE
import experimaestro_ir.metrics as metrics

@param('batch_size', default=64)
@param('device', type=Device, default=DEFAULT_DEVICE)
@param('preload', default=False)
@param('run_threshold', default=0, help="Maximum number of results (0 for not limit)")
@param('source', default='run')
@config()
class Reranker(predictors.BasePredictor):
    name = None

    def initialize(self, base_path: Path, measures: List[str], random, ranker, dataset):
        """Initialize a re-ranker

        Args:
            base_path: The base_path where file will be cached
            measures: List of measures to evaluate when running the re-ranker
            random (Random): Random number generator
            ranker (Ranker): The underlying ranker
            dataset (Dataset): The dataset on which to operate
        """
        self.base_path = base_path
        self.measures = measures
        self.ranker = ranker
        self.input_spec = self.ranker.input_spec()
        self.logger = Logger(self.__class__.__name__)
        self._device = self.device(self.logger)
        self.dataset = dataset
        self.random = random

    def _iter_batches(self, device):
        fields = set(self.input_spec['fields']) | {'query_id', 'doc_id'}
        it = datasets.record_iter(self.dataset,
                                  fields=fields,
                                  source=self.source,
                                  run_threshold=self.run_threshold,
                                  minrel=None,
                                  shuf=False,
                                  random=self.random,
                                  inf=False)
        for batch_items in util.chunked(it, self.batch_size):
            batch = {}
            for record in batch_items:
                for k, seq in record.items():
                    batch.setdefault(k, []).append(seq)
            batch = spec.apply_spec_batch(batch, self.input_spec, device)
            # ship 'em
            yield batch

    def _preload_batches(self, device):
        with self.logger.duration('loading evaluation data'):
            batches = list(self.logger.pbar(self._iter_batches(device), desc='preloading eval data (batches)'))
        while True:
            yield batches

    def _reload_batches(self):
        while True:
            it = self._iter_batches(self._device)
            yield it

    def pred_ctxt(self):
        if self.preload:
            datasource = self._preload_batches()
        else:
            datasource = self._reload_batches()


        return PredictorContext(self, datasource, self._device)

    def iter_scores(self, ranker, datasource, device):
        if isinstance(ranker, Trivial) and not ranker.neg and not ranker.qsum and not ranker.max:
            for qid, values in self.dataset.run().items():
                for did, score in values.items():
                    yield qid, did, score
            return
        if isinstance(ranker, Trivial) and not ranker.neg and not ranker.qsum and ranker.max:
            qrels = self.dataset.qrels()
            for qid, values in self.dataset.run().items():
                q_qrels = qrels.get(qid, {})
                for did in values:
                    yield qid, did, q_qrels.get(did, -1)
            return
        with torch.no_grad():
            ranker.eval()
            ds = next(datasource, None)
            total = None
            if isinstance(ds, list):
                total = sum(len(d['query_id']) for d in ds)
            elif self.source == 'run':
                if self.run_threshold > 0:
                    total = sum(min(len(v), self.run_threshold) for v in self.dataset.run().values())
                else:
                    total = sum(len(v) for v in self.dataset.run().values())
            elif self.source == 'qrels':
                total = sum(len(v) for v in self.dataset.qrels().values())
            with self.logger.pbar_raw(total=total, desc='pred', quiet=True) as pbar:
                for batch in util.background(ds):
                    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                    rel_scores = self.ranker(**batch).cpu()
                    if len(rel_scores.shape) == 2:
                        rel_scores = rel_scores[:, 0]
                    triples = list(zip(batch['query_id'], batch['doc_id'], rel_scores))
                    for qid, did, score in triples:
                        yield qid, did, score.item()
                    pbar.update(len(batch['query_id']))

    def rerank_dict(self, ranker, device):
        datasource = self._reload_batches(device)
        result = {}
        for qid, did, score in self.iter_scores(ranker, datasource, device):
            result.setdefault(qid, {})[did] = score
        return result


class PredictorContext:
    def __init__(self, pred, datasource, device):
        self.pred = pred
        self.datasource = datasource
        self.device = device

    def __call__(self, ctxt):
        cached = True
        epoch = ctxt['epoch']
        base_path = str(self.pred.base_path)
        os.makedirs(os.path.join(base_path, 'runs'), exist_ok=True)
        run_path = os.path.join(base_path, 'runs', f'{epoch}.run')
        if os.path.exists(run_path):
            # Use cached run file
            run = trec.read_run_dict(run_path)
        else:
            if self.pred.source == 'run' and self.pred.run_threshold > 0:
                official_run = self.pred.dataset.run('dict')
            else:
                official_run = {}
            run = {}
            ranker = ctxt['ranker']().to(self.device)
            this_qid = None 
            these_docs = {}
            with util.finialized_file(run_path, 'wt') as f:
                for qid, did, score in self.pred.iter_scores(ranker, self.datasource, self.device):
                    if qid != this_qid:
                        if this_qid is not None:
                            these_docs = self._apply_threshold(these_docs, official_run.get(this_qid, {}))
                            trec.write_run_dict(f, {this_qid: these_docs})
                        this_qid = qid
                        these_docs = {}
                    these_docs[did] = score
                if this_qid is not None:
                    these_docs = self._apply_threshold(these_docs, official_run.get(this_qid, {}))
                    trec.write_run_dict(f, {this_qid: these_docs})
            cached = False

        result = {
            'epoch': epoch,
            'run': run,
            'run_path': run_path,
            'base_path': base_path,
            'cached': cached
        }

        result['metrics'] = {m: None for m in self.pred.measures}
        result['metrics_by_query'] = {m: None for m in result['metrics']}

        missing_metrics = self.load_metrics(result)

        if missing_metrics:
            measures = set(missing_metrics)
            result['cached'] = False
            qrels = self.pred.dataset.qrels()
            calculated_metrics = metrics.calc(qrels, run_path, measures)
            result['metrics_by_query'].update(calculated_metrics)
            result['metrics'].update(metrics.mean(calculated_metrics))
            self.write_missing_metrics(result, missing_metrics)

        try:
            if ctxt['ranker']().add_runscore:
                result['metrics']['runscore_alpha'] = torch.sigmoid(ctxt['ranker']().runscore_alpha).item()
                rs_alpha_f = os.path.join(ctxt['base_path'], 'runscore_alpha.txt')
                with open(rs_alpha_f, 'at') as f:
                    plaintext.write_tsv(rs_alpha_f, [(str(epoch), str(result['metrics']['runscore_alpha']))])
        except FileNotFoundError:
            pass # model may no longer exist, ignore

        return result

    def load_metrics(self, ctxt):
        missing = set()
        epoch = ctxt['epoch']
        for metric in list(ctxt['metrics']):
            path_agg = os.path.join(ctxt['base_path'], metric, 'agg.txt')
            path_epoch = os.path.join(ctxt['base_path'], metric, f'{epoch}.txt')
            if os.path.exists(path_agg) and os.path.exists(path_epoch):
                ctxt['metrics'][metric] = [float(v) for k, v in plaintext.read_tsv(path_agg) if int(k) == epoch][0]
                ctxt['metrics_by_query'][metric] = {k: float(v) for k, v in plaintext.read_tsv(path_epoch)}
            else:
                missing.add(metric)
        return missing

    def write_missing_metrics(self, ctxt, missing_metrics):
        epoch = ctxt['epoch']
        for metric in missing_metrics:
            os.makedirs(os.path.join(ctxt['base_path'], metric), exist_ok=True)
            path_agg = os.path.join(ctxt['base_path'], metric, 'agg.txt')
            path_epoch = os.path.join(ctxt['base_path'], metric, f'{epoch}.txt')
            with open(path_agg, 'at') as f:
                plaintext.write_tsv(f, [(str(epoch), str(ctxt['metrics'][metric]))])
            plaintext.write_tsv(path_epoch, ctxt['metrics_by_query'][metric].items())

    def _apply_threshold(self, these_docs, original_scores):
        min_score = min(these_docs.values())
        missing_docs = original_scores.keys() - these_docs.keys()
        for i, did in enumerate(sorted(missing_docs, key=lambda did: original_scores[did], reverse=True)):
            these_docs[did] = min_score - i - 1
        return these_docs
