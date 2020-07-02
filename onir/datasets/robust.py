import os
from pytools import memoize_method
from functools import lru_cache
from experimaestro import task, config, param, pathoption, progress, configmethod, Choices, cache
from experimaestro_ir.anserini import Index as AnseriniIndex
from datamaestro_text.data.ir.trec import TrecAdhocAssessments, TrecAdhocTopics
from onir import datasets, util, indices, vocab
from .index_backed import IndexBackedDataset, Dataset, AssessedTopics
from onir.indices.sqlite import DocStore
from onir.interfaces import trec, plaintext
from datamaestro import prepare_dataset

# from <https://github.com/faneshion/DRMM/blob/9d348640ef8a56a8c1f2fa0754fe87d8bb5785bd/NN4IR.cpp>
FOLDS = {
    'f1': {'302', '303', '309', '316', '317', '319', '323', '331', '336', '341', '356', '357', '370', '373', '378', '381', '383', '392', '394', '406', '410', '411', '414', '426', '428', '433', '447', '448', '601', '607', '608', '612', '617', '619', '635', '641', '642', '646', '647', '654', '656', '662', '665', '669', '670', '679', '684', '690', '692', '700'},
    'f2': {'301', '308', '312', '322', '327', '328', '338', '343', '348', '349', '352', '360', '364', '365', '369', '371', '374', '386', '390', '397', '403', '419', '422', '423', '424', '432', '434', '440', '446', '602', '604', '611', '623', '624', '627', '632', '638', '643', '651', '652', '663', '674', '675', '678', '680', '683', '688', '689', '695', '698'},
    'f3': {'306', '307', '313', '321', '324', '326', '334', '347', '351', '354', '358', '361', '362', '363', '376', '380', '382', '396', '404', '413', '415', '417', '427', '436', '437', '439', '444', '445', '449', '450', '603', '605', '606', '614', '620', '622', '626', '628', '631', '637', '644', '648', '661', '664', '666', '671', '677', '685', '687', '693'},
    'f4': {'320', '325', '330', '332', '335', '337', '342', '344', '350', '355', '368', '377', '379', '387', '393', '398', '402', '405', '407', '408', '412', '420', '421', '425', '430', '431', '435', '438', '616', '618', '625', '630', '633', '636', '639', '649', '650', '653', '655', '657', '659', '667', '668', '672', '673', '676', '682', '686', '691', '697'},
    'f5': {'304', '305', '310', '311', '314', '315', '318', '329', '333', '339', '340', '345', '346', '353', '359', '366', '367', '372', '375', '384', '385', '388', '389', '391', '395', '399', '400', '401', '409', '416', '418', '429', '441', '442', '443', '609', '610', '613', '615', '621', '629', '634', '640', '645', '658', '660', '681', '694', '696', '699'}
}

_ALL = set.union(*FOLDS.values())
_FOLD_IDS = list(sorted(FOLDS.keys()))
for i in range(len(FOLDS)):
    FOLDS['tr' + _FOLD_IDS[i]] = _ALL - FOLDS[_FOLD_IDS[i]] - FOLDS[_FOLD_IDS[i-1]]
    FOLDS['va' + _FOLD_IDS[i]] = FOLDS[_FOLD_IDS[i-1]]
FOLDS['all'] = _ALL


@param("topics")
@param("assessments")
@config()
class TrecAssessedTopics(AssessedTopics):
    @cache("qrels.tsv")
    def qrels(self, fold_qrels_file):
        # Save the assessments
        with self.assessments.path.open("r") as fp:
            all_qrels = trec.read_qrels_dict(fp)
        fold_qrels = {qid: dids for qid, dids in all_qrels.items() if qid in self.qids}
        trec.write_qrels_dict(fold_qrels_file, fold_qrels)

    @cache("topics.tsv")
    def topics(self, path_topics):
        # Save the qrels; TODO: filter topics
        with util.finialized_file(path_topics, 'wt') as f, self.topics.path.open("rt") as query_file_stream:
            plaintext.write_tsv(f, trec.parse_query_format(query_file_stream))

@param("fold", default=None, checker=Choices(list(FOLDS.keys())))
@config()
class RobustAssessedTopics(TrecAssessedTopics):
    def __init__(self):
        self.qids = FOLDS[self.fold]

class IndexReader():
    def _init_iter_collection(self):
        # Using the trick here from capreolus, pulling document content out of public index:
        # <https://github.com/capreolus-ir/capreolus/blob/d6ae210b24c32ff817f615370a9af37b06d2da89/capreolus/collection/robust04.yaml#L15>
        index = indices.AnseriniIndex(self.index.path)
        total = index.num_docs()
        for ix, did in enumerate(self.logger.pbar(index.docids(), desc='documents')):
            progress(ix/total)
            raw_doc = index.get_raw(did)
            yield indices.RawDoc(did, raw_doc)


@param('index', type=AnseriniIndex, help="Index containing raw documents")
@pathoption("output_path", "docstore")
@task()
class BuildDocStore(IndexReader):
    def execute(self):
        idxs = [indices.SqliteDocstore(self.path_docs)]
        self._init_indices_parallel(idxs, self._init_iter_collection(), True)
 

@param('index', type=AnseriniIndex, help="Index containing raw documents")
@pathoption("output_path", "index")
@task()
class Reindex(IndexReader):
    def execute(self):
        idxs = [indices.AnseriniIndex(self.output_path, stemmer='none')]
        self._init_indices_parallel(idxs, self._init_iter_collection(), True)

# # TODO: Run Reindex and BuildDocStore in parallel
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



@param('index_stem', type=AnseriniIndex)
@param('index', type=AnseriniIndex)
@param('docstore', type=DocStore)
@task()
class RobustDataset(Dataset):
    """Prepares the Robust dataset from a pre-computed index"""
    def __init__(self):
        super().__init__(self)
        self.index = indices.AnseriniIndex(self.index.path)
        self.index_stem = indices.AnseriniIndex(self.index_stem.path)
        self.doc_store = indices.SqliteDocstore(self.docstore.path)

    @configmethod
    @staticmethod
    def prepare(**kwargs):
        return RobustDatasetGenerator()

    def _get_index(self, record):
        return self.index

    def _get_docstore(self):
        return self.doc_store

    def _get_index_for_batchsearch(self):
        return self.index_stem

    @memoize_method
    def _load_queries_base(self, subset):
        topics = self._load_topics()
        result = {}
        for qid in FOLDS[subset]:
            result[qid] = topics[qid]
        return result

    def qrels(self, fmt='dict'):
        return self._load_qrels(self.subset, fmt)

    @memoize_method
    def _load_qrels(self, subset, fmt):
        return trec.read_qrels_fmt(str(self.assessments.path), fmt)

    @memoize_method
    def _load_topics(self):
        result = {}
        for item, qid, text in plaintext.read_tsv(str(self.path_topics)):
            if item == 'topic':
                result[qid] = text
        return result

class RobustDatasetGenerator:
    def __init__(self):
        self.index_stem = prepare_dataset("ca.uwaterloo.jimmylin.anserini.robust04")
        self.topics = prepare_dataset("gov.nist.trec.adhoc.robust.2004.topics")
        self.qrels = prepare_dataset("gov.nist.trec.adhoc.robust.2004.qrels")
        self.docstore = BuildDocStore(index=self.index_stem).submit()
        self.index = Reindex(index=self.index_stem).submit()

    def subset(self, fold: str, **kwargs):
        """Returns dataset with a specific query fold
        """

        topics = RobustAssessedTopics(topics=self.topics, assessments=self.qrels, fold=fold)
        return RobustDataset(index_stem=self.index_stem, index=self.index, assessed_topics=topics, **kwargs)