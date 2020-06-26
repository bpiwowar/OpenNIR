from experimaestro import config, param
from onir import util, vocab, log

@config()
class Dataset:
    def __init__(self):
        self.logger = log.Logger(self.__class__.__name__)

    def initialize(self, vocab):
        self.vocab = vocab

    def qrels(self, fmt='dict'):
        raise NotImplementedError

    def run(self, fmt='dict'):
        raise NotImplementedError

    def build_record(self, fields, **initial_values):
        raise NotImplementedError

    def all_doc_ids(self):
        raise NotImplementedError

    def num_docs(self):
        raise NotImplementedError

    def all_query_ids(self):
        raise NotImplementedError

    def all_queries_raw(self):
        raise NotImplementedError

    def num_queries(self):
        raise NotImplementedError

    def _confirm_dua(self):
        if self._has_confirmed_dua is None and self.DUA is not None:
            self._has_confirmed_dua = util.confirm(self.DUA.format(ds_path=util.path_dataset(self)))
        return self._has_confirmed_dua
