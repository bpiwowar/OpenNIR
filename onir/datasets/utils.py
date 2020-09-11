from experimaestro import config, param, cache
from onir.interfaces import trec, plaintext
from onir import util

@config()
class AssessedTopics():
    """Abstract class encapsulating topics with their associated relevance assessments"""
    pass

@param("topics")
@param("assessments")
@config()
class TrecAssessedTopics(AssessedTopics):
    """Asssessed topics - supposes that files are following the MS Marco format:
    
    - assessments are in the standard TREC format
    
    """

    @cache("qrels.tsv")
    def qrels_path(self, fold_qrels_file):
        if not fold_qrels_file.is_file():
            with self.assessments.path.open("r") as fp:
                all_qrels = trec.read_qrels_dict(fp)
            fold_qrels = {qid: dids for qid, dids in all_qrels.items() if qid in self.qids}
            trec.write_qrels_dict(fold_qrels_file, fold_qrels)
        return fold_qrels_file

    @cache("topics.tsv")
    def topics_path(self, path_topics):
        # Save the topics
        if not path_topics.is_file():
            with util.finialized_file(path_topics, 'wt') as f, self.topics.path.open("rt") as query_file_stream:
                data = ((item, qid, text) for item, qid, text in trec.parse_query_format(query_file_stream) if qid in self.qids)
                plaintext.write_tsv(f, data)
        return path_topics
