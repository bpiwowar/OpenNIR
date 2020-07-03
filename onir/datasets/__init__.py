from onir.datasets.base import Dataset

# Default iteration functions over datasets
from onir.datasets.query_iter import QueryIter as query_iter
from onir.datasets.doc_iter import DocIter as doc_iter
from onir.datasets.pair_iter import pair_iter
from onir.datasets.record_iter import record_iter, run_iter, qrels_iter, pos_qrels_iter
from onir.datasets.utils import AssessedTopics, TrecAssessedTopics
