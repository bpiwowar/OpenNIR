import logging
import os
from pathlib import Path

from datamaestro import prepare_dataset
from experimaestro.click import click, forwardoption
from experimaestro import experiment, tag
from onir import rankers
from onir.datasets.robust import RobustDataset
from onir.predictors.reranker import Reranker, Device
from onir.random import Random
from onir.tasks.learner import Learner
from onir.tasks.evaluate import Evaluate
from onir.trainers.pointwise import PointwiseTrainer
from onir.vocab.wordvec_vocab import WordvecUnkVocab
from experimaestro_ir.models import BM25
from experimaestro_ir.anserini import SearchCollection
from experimaestro_ir.evaluation import TrecEval
import importlib
from typing import List, Callable, NamedTuple

logging.basicConfig(level=logging.INFO)



# --- Experiment

@forwardoption.max_epoch(Learner)
@click.option("--debug", is_flag=True, help="Print debug information")
@click.option("--gpu", is_flag=True, help="Use GPU")
@click.option("--port", type=int, default=12345, help="Port for monitoring")
@click.argument("workdir", type=Path)
@click.group(chain=True, invoke_without_command=False)
def cli(**kwargs): pass

class Information:
    vocab = None
    device = None

    datasets = []
    rankers = []

def register(method, add):
    def m(**kwargs): 
        return lambda info: add(info, method(info, **kwargs))
    m.__name__ = method.__name__
    m.__doc__ = method.__doc__
    return cli.command()(m)


# ---- Datasets

def dataset(method):
    return register(method, lambda info, ds: info.datasets.append(ds))

@dataset
def msmarco(info):
    """Use the MS Marco dataset"""
    logging.info("Adding MS Marco dataset")
    from onir.datasets.msmarco import MsmarcoDataset
    ds = MsmarcoDataset.prepare()
    return ds("train"), ds("dev"), ds("trec2019.test")


@dataset
def robust(info):
    """Use the TREC Robust dataset"""
    from onir.datasets.robust import RobustDataset
    ds = RobustDataset.prepare()
    return ds("trf1"), ds("vaf1"), ds("f1")

# ---- Vocabulary

def vocab(info):
    return register(method, lambda info, vocab: info.vocab = vocab)

@vocab
def glove(info):
    wordembs = prepare_dataset("edu.stanford.glove.6b.50")        
    return WordvecUnkVocab(data=wordembs, random=random)

@vocab
def bertvocab(info):
    wordembs = prepare_dataset("edu.stanford.glove.6b.50")        
    return WordvecUnkVocab(data=wordembs, random=random)

# ---- Models

def model(method):
    return register(method, lambda info, model: info.rankers.append(model))

@model
def drmm(info):
    """Use the DRMM model"""
    return rankers.Drmm(vocab=info.vocab).tag("model", "drmm")

@model
def vanilla_transformer(info):
    """Use the Vanilla BERT model"""
    from onir.rankers.vanilla_transformer import VanillaTransformer
    return VanillaTransformer(vocab=info.vocab).tag("model", "vanilla-transformer")

# --- Run the experiment

@cli.resultcallback()
def process(processors, debug, gpu, port, workdir, max_epoch):
    """Runs an experiment"""
    logging.info("Running pipeline")

    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)
    info = Information()
    info.device = device = Device(gpu=gpu)
    
    # Sets the working directory and the name of the xp
    with experiment(workdir, "neural-ir", port=port) as xpm:
        # Misc settings
        random = Random()
        assert "JAVA_HOME" in os.environ, "JAVA_HOME should be defined (to call anserini)"
        xpm.setenv("JAVA_HOME", os.environ["JAVA_HOME"])

        # Prepare the embeddings
        wordembs = prepare_dataset("edu.stanford.glove.6b.50")        
        info.device = device
        info.vocab = WordvecUnkVocab(data=wordembs, random=random)

        for processor in processors:
            processor(info)

        assert info.datasets, "No dataset was selected"
        assert info.rankers, "No model was selected"

        for train, val, test in info.datasets:

            # Search and evaluate with BM25
            bm25_search = (
                SearchCollection(index=test.index, topics=test.assessed_topics.topics, model=BM25())
                .tag("model", "bm25")
                .submit()
            )
            bm25_eval = TrecEval(
                assessments=test.assessed_topics.assessments, run=bm25_search
            ).submit()

            # Train and evaluate with each model
            for ranker in info.rankers:
                # Train with OpenNIR DRMM model
                predictor = Reranker(device=device)
                trainer = PointwiseTrainer(device=device)
                learner = Learner(trainer=trainer, random=random, ranker=ranker, valid_pred=predictor, 
                    train_dataset=train, val_dataset=val, max_epoch=tag(max_epoch))
                model = learner.submit()

                # Evaluate the neural model
                evaluate = Evaluate(dataset=test, model=model, predictor=predictor).submit()

        xpm.wait()
    
        print(f"Results for DRMM\n{evaluate.results.read_text()}\n")
        print(f"Results for BM25\n{bm25_eval.results.read_text()}\n")

if __name__ == "__main__":
    cli()

