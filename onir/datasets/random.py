from experimaestro import param, config, option
from onir import datasets

@param('qlen', default=5)
@param('dlen', default=500)
@param('count', default=10000)

@config()
class RandomDataset(datasets.Dataset):
    """
    Dataset producing random samples, used for controlled tests in scripts/perf_benchmark.py
    """

    def __init__(self, config, logger, vocab, random):
        super().__init__(config, logger, vocab)
        self.random = random

    def run(self, fmt="dict"):
        return None

    def qrels(self, fmt="dict"):
        return None

    def all_doc_ids(self):
        return []

    def all_query_ids(self):
        return []

    def build_record(self, fields, **initial_values):
        result = dict(initial_values)
        for field in sorted(fields):
            l = 1
            if field.startswith('query_'):
                l = self.qlen
            elif field.startswith('doc_'):
                l = self.dlen
            if field in ('runscore', 'relscore'):
                result[field] = self.random.rand()
            elif field.endswith('_len'):
                result[field] = l
            elif field.endswith('_tok'):
                result[field] = list(self.random.randint(1, self.vocab.lexicon_size(), size=l))
            elif field.endswith('_idf') or field.endswith('_score'):
                result[field] = list(self.random.rand(l))
            else:
                raise ValueError(f'unsupported field: {field}')
        return result

    def record_iter(self,
                    fields: set,
                    source: str, # one of ['run', 'qrels']
                    minrel: None, # integer indicating the minimum relevance score, or None for unfiltered
                    shuf: bool = True,
                    random=None,
                    inf: bool = False):
        self.logger.warn(f'source={source} minrel={minrel} and shuf={shuf} do not apply to RandomDataset')
        first = True
        while first or inf:
            first = False
            for _ in range(self.count):
                yield self.build_record(fields)
