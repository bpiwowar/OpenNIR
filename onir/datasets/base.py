from experimaestro import config, param
from onir import util, vocab, log

@param("vocab", type=vocab.Vocab)
@config()
class Dataset:
    name = None
    DUA = None

    def __initialize__(self):
        self.logger = log.Logger(self.__class__.__name__)

    def qrels(self, fmt='dict'):
        raise NotImplementedError

    def run(self, fmt='dict'):
        raise NotImplementedError

    def path_segment(self):
        raise NotImplementedError

    def collection_path_segment(self):
        raise NotImplementedError

    def build_record(self, fields, **initial_values):
        raise NotImplementedError

    def all_doc_ids(self):
        raise NotImplementedError

    def num_docs(self):
        raise NotImplementedError

    def all_query_ids(self):
        raise NotImplementedError

    def num_queries(self):
        raise NotImplementedError

    def lexicon_path_segment(self):
        return self.path_segment() + '_' + self.vocab.lexicon_path_segment()

    def _confirm_dua(self):
        if self._has_confirmed_dua is None and self.DUA is not None:
            self._has_confirmed_dua = util.confirm(self.DUA.format(ds_path=util.path_dataset(self)))
        return self._has_confirmed_dua
