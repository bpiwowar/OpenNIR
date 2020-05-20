import json
from experimaestro import task, param, progress
from onir.datasets import Dataset
from .learner import Learner

@param("dataset", type=Dataset)
@param("model", type=Learner)
@param('metrics', default='map,p@20,ndcg')
@task()
class Evaluate:
    def execute(self):
        # Load top train context
        with open(model.valtest_path, "r") as fp:
            data = json.load(fp)

        data[""] = top_train_ctxt    
        top_train_ctxt['ranker'] = onir.trainers.base._load_ranker(top_train_ctxt['ranker'](), top_train_ctxt['ranker_path'])

        with self.logger.duration('testing'):
            test_ctxt = self.test_pred.run(top_train_ctxt)

        file_output.update({
            'test_ds': self.test_pred.dataset.path_segment(),
            'test_run': test_ctxt['run_path'],
            'test_metrics': test_ctxt['metrics'],
        })

        self.logger.info('test run at {}'.format(test_ctxt['run_path']))
        self.logger.info('valid ' + self._build_valid_msg(top_valid_ctxt))
        self.logger.info('test  ' + self._build_valid_msg(test_ctxt))

        raise NotImplementedError("Should finish")
