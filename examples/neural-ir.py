import logging
import os
from pathlib import Path

from datamaestro import prepare_dataset
from experimaestro.click import click, forwardoption
from experimaestro import experiment
from onir import rankers
from onir.datasets.robust import RobustDataset
from onir.predictors.reranker import Reranker, Device
from onir.random import Random
from onir.tasks.learner import Learner
from onir.tasks.evaluate import Evaluate
from onir.trainers.pointwise import PointwiseTrainer
from onir.vocab.wordvec_vocab import WordvecUnkVocab

logging.basicConfig(level=logging.INFO)


# --- Defines the experiment

@forwardoption.max_epoch(Learner)
@click.option("--debug", is_flag=True, help="Print debug information")
@click.option("--gpu", is_flag=True, help="Use GPU")
@click.option("--port", type=int, default=12345, help="Port for monitoring")
@click.argument("workdir", type=Path)
@click.command()
def cli(port, gpu, workdir, debug, max_epoch):
    """Runs an experiment"""
    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)
    device = Device(gpu=gpu)
    
    # Sets the working directory and the name of the xp
    with experiment(workdir, "drmm", port=port) as xp:
        # Prepare the collection
        random = Random()
        robust = RobustDataset.prepare().submit()
        xp.setenv("JAVA_HOME", os.environ["JAVA_HOME"])
        
        # Prepare the embeddings
        wordembs = prepare_dataset("edu.stanford.glove.6b.50")        
        vocab = WordvecUnkVocab(data=wordembs, random=random)

        # Train with OpenNIR DRMM model
        ranker = rankers.Drmm(vocab=vocab).tag("ranker", "drmm")
        predictor = Reranker(device=device)
        trainer = PointwiseTrainer(device=device)
        learner = Learner(trainer=trainer, random=random, ranker=ranker, valid_pred=predictor, 
            train_dataset=robust.subset('trf1'), val_dataset=robust.subset('vaf1'), max_epoch=max_epoch)
        model = learner.submit()

        # Evaluate
        Evaluate(dataset=robust.subset('f1'), model=model, predictor=predictor).submit()


if __name__ == "__main__":
    cli()
