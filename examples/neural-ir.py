import logging
import os
from pathlib import Path

import click

from datamaestro import prepare_dataset
from experimaestro import experiment
from onir.datasets.robust import RobustDataset
from onir.predictors.reranker import Reranker
from onir.random import Random
from onir.rankers.drmm import Drmm
from onir.tasks.learner import Learner
from onir.trainers.pointwise import PointwiseTrainer
from onir.vocab.wordvec_vocab import WordvecUnkVocab

logging.basicConfig(level=logging.INFO)



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
        learner = Learner(trainer=trainer, random=random, ranker=ranker, valid_pred=predictor, 
            train_dataset=robust.subset('trf1'), val_dataset=robust.subset('vaf1'))
        if small:
            learner.max_epoch = 2
        model = learner.submit()

        # Evaluate
        # Evaluate(dataset=robust.subset('f1'), model=model).submit()


if __name__ == "__main__":
    cli()
