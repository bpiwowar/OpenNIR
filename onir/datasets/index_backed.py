import os
import functools
import contextlib
from glob import glob
import pickle
from experimaestro import param, config, cache, configmethod, pathoption, progress
from experimaestro_ir.anserini import IndexCollection
from onir import datasets, util, log, indices
from onir.interfaces import trec
from onir.datasets import AssessedTopics
from onir.indices.sqlite import DocStore
from experimaestro_ir.anserini import Index as AnseriniIndex
from experimaestro import task, config

def _init_indices_parallel(indices, doc_iter, force):
    """Builds the indices

    Args:
        indices (List[onir.indices.BaseIndex]): The indices to build
        doc_iter (Iterator[onir.indices.misc.RawDoc]): [description]
        force (bool): Whether indices should be rebuilt
    """
    needs_docs = []
    for index in indices:
        if force or not index.built():
            needs_docs.append(index)

    if needs_docs:
        with contextlib.ExitStack() as stack:
            doc_iters = util.blocking_tee(doc_iter, len(needs_docs))
            for idx, it in zip(needs_docs, doc_iters):
                stack.enter_context(util.CtxtThread(functools.partial(idx.build, it)))

def _iter_collection(index):
    # Using the trick here from capreolus, pulling document content out of public index:
    # <https://github.com/capreolus-ir/capreolus/blob/d6ae210b24c32ff817f615370a9af37b06d2da89/capreolus/collection/robust04.yaml#L15>
    index = indices.AnseriniIndex(index.path)
    total = index.num_docs()
    logger = log.easy()
    for ix, did in enumerate(logger.pbar(index.docids(), desc='documents')):
        progress(ix/total)
        raw_doc = index.get_raw(did)
        yield indices.RawDoc(did, raw_doc)

@param('index', type=AnseriniIndex, help="Index containing raw documents")
@pathoption("path", "docstore")
@task()
class BuildDocStore(DocStore):
    """Build doc-store from index"""
    def execute(self):
        idxs = [indices.SqliteDocstore(self.path)]
        _init_indices_parallel(idxs, _iter_collection(self.index), True)
 
@param('index', type=AnseriniIndex, help="Index containing raw documents")
@pathoption("path", "index")
@task()
class Reindex(AnseriniIndex):
    """Build index (not stemmed) from index"""
    def execute(self):
        idxs = [indices.AnseriniIndex(self.path, stemmer='none')]
        _init_indices_parallel(idxs, _iter_collection(self.index), True)


# TODO: Run Reindex and BuildDocStore in parallel when @multitask is implemented in experimaestro
# @param('index', type=AnseriniIndex, help="Index containing raw documents")
# @multitask
# class Prepare(IndexReader):
#     def __outputs__(self):
#         return [ Reindex(index=self.index), BuildDocStore(index=self.index) ]

#     def execute(self):
#         idxs = []
#         if self.docstore:
#             idxs.append(self.docstore.idx())
#         if self.reindex:
#             idxs.append(self.reindex.idx())
#         self._init_indices_parallel(idxs, self._init_iter_collection(), True)




@param('rankfn', default='bm25')
@param('subset', default='all')
@param('ranktopk', default=1000)
@param('assessed_topics', type=AssessedTopics)
@config()
class Dataset(datasets.Dataset):
    """
    Dataset base class for using an (Anserini) index as the source of the data.
    """

    def __init__(self):
        super().__init__()
        self.run_cache = {}
        self.logger = log.easy()

    def build_record(self, fields, **initial_values):
        record = LazyDataRecord(self, **initial_values)
        record.load(fields)
        return record

    def run(self, fmt='dict'):
        return self._load_run_base(self._get_index_for_batchsearch(),
                                   self.rankfn,
                                   self.ranktopk,
                                   fmt=fmt)

    def run_dict(self):
        return self._load_run_base(self._get_index_for_batchsearch(),
                                   self.rankfn,
                                   self.ranktopk,
                                   fmt='dict')

    def all_doc_ids(self):
        yield from self._get_docstore().docids()

    def num_docs(self):
        return self._get_docstore().num_docs()

    def all_query_ids(self):
        yield from self._load_queries_base(self.subset).keys()

    def all_queries_raw(self):
        return self._load_queries_base(self.config['subset']).items()

    def num_queries(self):
        return sum(1 for _ in self.all_query_ids())

    @cache("runs")
    def _load_run_base(self, run_path, index, rankfn, ranktopk, fmt='dict', fscache=False, memcache=True):
        key = (index.path(), rankfn, ranktopk, fmt)
        if memcache and key in self.run_cache:
            return self.run_cache[key]

        run_path.mkdir(parents=True, exist_ok=True)
        run_path = run_path / f"{rankfn}.{ranktopk}.run"
        run_path_cache = f'{run_path}.{fmt}.cache'

        result = self._load_run_base_fscached(run_path_cache, rankfn, ranktopk, fscache)
        if result is None:
            result = self._load_run_base_direct(str(run_path), rankfn, ranktopk, fmt)
        if result is None:
            result = self._load_run_base_infer(str(run_path.parent), rankfn, ranktopk, fmt)
        if result is None:
            result = self._load_run_base_query(index, rankfn, ranktopk, str(run_path), fmt)

        if fscache and not os.path.exists(run_path_cache):
            with open(run_path_cache, 'wb') as f:
                with self.logger.duration(f'caching {rankfn}:{ranktopk}'):
                    pickle.dump(result, f)

        if memcache:
            self.run_cache[key] = result

        return result

    def _load_run_base_fscached(self, run_path_cache, rankfn, ranktopk, fscache=False):
        if fscache and os.path.exists(run_path_cache):
            with open(run_path_cache, 'rb') as f:
                with self.logger.duration(f'reading {rankfn}:{ranktopk} (cached)'):
                    return pickle.load(f)
        return None

    def _load_run_base_direct(self, run_path, rankfn, ranktopk, fmt='dict'):
        if os.path.exists(run_path):
            with self.logger.duration(f'reading {rankfn}:{ranktopk}'):
                return trec.read_run_fmt(run_path, fmt=fmt)
        return None

    def _load_run_base_infer(self, run_dir, rankfn, ranktopk, fmt='dict'):
        best_candidate_run, best_topk = None, None
        for candidate_run in glob(os.path.join(run_dir, f'{rankfn}.*.run')):
            c_topk = int(candidate_run.split('.')[-2])
            if c_topk > ranktopk and (best_topk is None or c_topk < best_topk):
                best_candidate_run = candidate_run
                best_topk = c_topk
        if best_candidate_run is not None:
            with self.logger.duration(f'loading {rankfn}:{ranktopk} from larger batch {rankfn}:{best_topk}'):
                return trec.read_run_fmt(best_candidate_run, fmt, top=ranktopk)
        return None

    def _load_run_base_query(self, index, rankfn, ranktopk, run_path, fmt):
        queries = self._load_queries_base().items()
        index.batch_query(queries, rankfn, ranktopk, destf=run_path)
        return trec.read_run_fmt(run_path, fmt)

    def _lang(self):
        return "en"

    def _query_rawtext(self, record):
        return self._load_queries_base()[record['query_id']]

    def _query_text(self, record):
        return tuple(self.vocab.tokenize(record['query_rawtext']))

    def _query_tok(self, record):
        return [self.vocab.tok2id(t) for t in record['query_text']]

    def _query_idf(self, record):
        index = self._get_index(record)
        return [index.term2idf(t) for t in record['query_text']]

    def _query_len(self, record):
        return len(record['query_text'])

    def _query_lang(self, record):
        return self._lang()

    def _query_score(self, record):
        index = self._get_index(record)
        return index.get_query_doc_scores(record['query_text'], record['doc_id'], self.rankfn)[1]

    def _doc_rawtext(self, record):
        docstore = self._get_docstore()
        return docstore.get_raw(record['doc_id'])

    def _doc_text(self, record):
        return self.vocab.tokenize(record['doc_rawtext'])

    def _doc_tok(self, record):
        return [self.vocab.tok2id(t) for t in record['doc_text']]

    def _doc_idf(self, record):
        index = self._get_index(record)
        return [index.term2idf(t) for t in record['doc_rawtext']]

    def _doc_len(self, record):
        return len(record['doc_text'])

    def _doc_lang(self, record):
        return self._lang()

    def _runscore(self, record):
        index = self._get_index(record)
        return index.get_query_doc_scores(record['query_text'], record['doc_id'], self.rankfn)[0]

    def _relscore(self, record):
        return float(self.qrels('dict').get(record['query_id'], {}).get(record['doc_id'], -999))

    def qrels(self, fmt='dict'):
        return self.qrels(fmt=fmt)


class LazyDataRecord:
    def __init__(self, ds, **data):
        # pylint: disable=W0212
        self.ds = ds
        self._data = data
        self.methods = {
            'query_rawtext': ds._query_rawtext,
            'query_text': ds._query_text,
            'query_tok': ds._query_tok,
            'query_idf': ds._query_idf,
            'query_len': ds._query_len,
            'query_lang': ds._query_lang,
            'query_score': ds._query_score,
            'doc_rawtext': ds._doc_rawtext,
            'doc_text': ds._doc_text,
            'doc_tok': ds._doc_tok,
            'doc_idf': ds._doc_idf,
            'doc_len': ds._doc_len,
            'doc_lang': ds._doc_lang,
            'runscore': ds._runscore,
            'relscore': ds._relscore,
        }

    def __getitem__(self, key):
        if key not in self._data:
            if key in self.methods:
                self._data[key] = self.methods[key](self)
            else:
                raise ValueError(f'Unsupported input `{key}`')
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def load(self, fields):
        return {f: self[f] for f in fields}
