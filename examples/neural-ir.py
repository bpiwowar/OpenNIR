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
from onir.trainers.pointwise import PointwiseTrainer
from onir.datasets.robust import RobustDataset
from onir.tasks.learner import Learner
from onir.random import Random
from onir.vocab.wordvec_vocab import WordvecUnkVocab
from onir.predictors.reranker import Reranker

# --- Defines the experiment


@click.option("--small", is_flag=True, help="Reduce the number of iterations (testing)")
@click.option("--debug", is_flag=True, help="Print debug information")
@click.option("--port", type=int, default=12345, help="Port for monitoring")
@click.argument("workdir", type=Path)
@click.command()
def cli(port, workdir, debug, small):
    """Runs an experiment"""
    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)

    # Sets the working directory and the name of the xp
    with experiment(workdir, "drmm", port=port) as xp:
        xp.setenv("JAVA_HOME", os.environ["JAVA_HOME"])
        
        # Prepare the collection
        wordembs = prepare_dataset("edu.stanford.glove.6b.50")        
        random = Random()
        vocab = WordvecUnkVocab(data=wordembs, random=random)
        robust = RobustDataset.prepare().submit()

        # Train with OpenNIR DRMM model
        ranker = Drmm(vocab=vocab).tag("ranker", "drmm")
        predictor = Reranker()
        trainer = PointwiseTrainer()
        learner = Learner(trainer=trainer, random=random, ranker=ranker, valid_pred=predictor, dataset=robust)
        if small:
            learner.max_epoch = 2
        learner.submit()


if __name__ == "__main__":
    cli()
