import os
from pytools import memoize_method
from experimaestro import task, param, pathargument
from experimaestro_ir.anserini import Index as AnseriniIndex
from datamaestro_text.data.trec import TrecAdhocAssessments, TrecAdhocTopics
from onir import datasets, util, indices, vocab
from .index_backed import IndexBackedDataset
from onir.interfaces import trec, plaintext

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

@param('subset', default='all')
@param('ranktopk', default=100)

@param('anserini_index', type=AnseriniIndex)
@param('queries', type=TrecAdhocTopics)
@param('assessments', type=TrecAdhocAssessments)

@pathargument("path_anserini", "anserini")
@pathargument("path_anserini_porter", "anserini.porter")
@pathargument("path_docs", "docs.sqlite")
@pathargument("path_folds", "folds")
@pathargument("path_topics", "topics")
@task()
class RobustDataset(IndexBackedDataset):
    """
    Interface to the TREC Robust 2004 dataset.
     > Ellen M. Voorhees. 2004. Overview of TREC 2004. In TREC.
    """

    def __initialize__(self):
        IndexBackedDataset.__initialize__(self)
        self.index = indices.AnseriniIndex(self.path_anserini, stemmer='none')
        self.index_stem = indices.AnseriniIndex(self.path_anserini_porter, stemmer='porter')
        self.doc_store = indices.SqliteDocstore(self.path_docs)

    @staticmethod
    def prepare(**kwargs):
        from datamaestro import prepare_dataset
        index = prepare_dataset("ca.uwaterloo.jimmylin.anserini.robust04")
        qrels = prepare_dataset("gov.nist.trec.adhoc.robust.2004.qrels")
        topics = prepare_dataset("gov.nist.trec.adhoc.robust.2004.topics")
        
        return RobustDataset(anserini_index=index, assessments=qrels, queries=topics, **kwargs)

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

    def execute(self):
        idxs = [self.index, self.index_stem, self.doc_store]
        self._init_indices_parallel(idxs, self._init_iter_collection(), True)

        for fold in FOLDS:
            fold_qrels_file = self.path_folds.with_suffix(f".{fold}.qrels")
            with self.assessments.path.open("r") as fp:
                all_qrels = trec.read_qrels_dict(fp)
            fold_qrels = {qid: dids for qid, dids in all_qrels.items() if qid in FOLDS[fold]}
            trec.write_qrels_dict(fold_qrels_file, fold_qrels)

        with util.finialized_file(self.path_topics, 'wt') as f, self.queries.path.open("rt") as query_file_stream:
            plaintext.write_tsv(f, trec.parse_query_format(query_file_stream))

    def _init_iter_collection(self):
        # Using the trick here from capreolus, pulling document content out of public index:
        # <https://github.com/capreolus-ir/capreolus/blob/d6ae210b24c32ff817f615370a9af37b06d2da89/capreolus/collection/robust04.yaml#L15>
        index = indices.AnseriniIndex(self.anserini_index.path)
        for did in self.logger.pbar(index.docids(), desc='documents'):
            raw_doc = index.get_raw(did)
            yield indices.RawDoc(did, raw_doc)
