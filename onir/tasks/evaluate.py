import json
from experimaestro import task, param, progress, pathargument
from onir.datasets import Dataset
from onir import log, predictors
import numpy as np
from .learner import Learner

@param("dataset", type=Dataset)
@param("model", type=Learner)
@param("predictor", type=predictors.BasePredictor)
@param('metrics', default='map,p@20,ndcg')
@pathargument("detailed", "detailed.txt")
@pathargument("measures", "measures.txt")
@task()
class Evaluate:
    def execute(self):
        # Load top train context
        with open(self.model.valtest_path, "r") as fp:
            data = json.load(fp)

        self.logger = log.easy()

        random = np.random.RandomState() # TODO: should not be necessary when not training
        state_path = data["valid_path"]
        ranker = self.model.ranker
        ranker.initialize(random)
        ranker.load(state_path)

        self.predictor.initialize(random, ranker, self.dataset)


        self.dataset.initialize(ranker.vocab)

        with self.logger.duration('testing'):
            test_ctxt = self.predictor.run({
                "epoch": data["valid_epoch"],
                "ranker": lambda: ranker
            })

        def print_line(fp, measure, scope, value):
            fp.write('{:25s}{:8s}{:.4f}\n'.format(measure, scope, value))

        with open(self.measures, "wt") as fp:
            for measure, value in test_ctxt["metrics"].items():
                print_line(fp, measure, 'all', value)

        with open(self.detailed, "wt") as fp:        
            for measure, value in test_ctxt["metrics_by_query"].items():
                for query, value in value.items():
                    print_line(fp, measure, query, value)
