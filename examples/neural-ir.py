import click
from pathlib import Path
import os
from datamaestro import prepare_dataset
import logging
import multiprocessing


logging.basicConfig(level=logging.INFO)
CPU_COUNT = multiprocessing.cpu_count()


from experimaestro import experiment
from experimaestro_ir.evaluation import TrecEval
from experimaestro_ir.models import BM25
from experimaestro_ir.anserini import IndexCollection, SearchCollection

from onir.rankers.drmm import Drmm
from onir.trainers import PointwiseTrainer
from onir.datasets.robust import RobustDataset
from onir.pipelines import Learner
from onir.random import Random
from onir.vocab import WordvecVocab
from onir.predictors import Reranker

# --- Defines the experiment


@click.option("--debug", is_flag=True, help="Print debug information")
@click.option("--port", type=int, default=12345, help="Port for monitoring")
@click.argument("workdir", type=Path)
@click.command()
def cli(port, workdir, debug):
    """Runs an experiment"""
    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)

    bm25 = BM25()

    # Sets the working directory and the name of the xp
    with experiment(workdir, "index", port=port) as xp:
        xp.setenv("JAVA_HOME", os.environ["JAVA_HOME"])
        # Index the collection
        wordembs = prepare_dataset("edu.stanford.glove.6b.50")        
        random = Random()
        vocab = WordvecVocab(data=wordembs)
        robust = RobustDataset.prepare(vocab).submit()


        # Train with OpenNIR DRMM model
        ranker = Drmm(random=random, vocab=vocab).tag("ranker", "drmm")
        predictor = Reranker(ranker=ranker)
        trainer = PointwiseTrainer(random=random, vocab=vocab, ranker=ranker, dataset=robust)
        learner = Learner(trainer=trainer, valid_pred=predictor).submit()

        # # search = ModelRerank(
        # #     base=bm25, topics=training_ds.topics, model=learnedmodel
        # # ).submit()
        # # eval = TrecEval(assessments=training_ds.assessments, results=search).submit()


if __name__ == "__main__":
    cli()
