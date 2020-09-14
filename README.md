# OpenNIR (experimaestro version)

OpenNIR-xpm is an end-to-end neural ad-hoc ranking pipeline.

This is an adaptation of [OpenNIR](https://github.com/Georgetown-IR-Lab/OpenNIR) using experiment manager tools (experimaestro and datamaestro).


## Quick start

Install OpenNIR (XPM version) with

```python3
pip install OpenNIR-XPM
```

and use this file for training (other examples are available on github)
```python
import logging
import os
from pathlib import Path

from datamaestro import prepare_dataset
from experimaestro.click import click, forwardoption
from experimaestro import experiment
from onir.datasets.robust import RobustDataset
from onir.predictors.reranker import Reranker
from onir.random import Random
from onir.rankers.drmm import Drmm
from onir.tasks.learner import Learner
from onir.tasks.evaluate import Evaluate
from onir.trainers.pointwise import PointwiseTrainer
from onir.vocab.wordvec_vocab import WordvecUnkVocab

logging.basicConfig(level=logging.INFO)


# --- Defines the experiment

@forwardoption.max_epoch(Learner)
@click.option("--debug", is_flag=True, help="Print debug information")
@click.option("--port", type=int, default=12345, help="Port for monitoring")
@click.argument("workdir", type=Path)
@click.command()
def cli(port, workdir, debug, max_epoch):
    """Runs an experiment"""
    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)

    # Sets the working directory and the name of the xp
    with experiment(workdir, "drmm", port=port) as xp:
        random = Random()
        xp.setenv("JAVA_HOME", os.environ["JAVA_HOME"])

        # Prepare the collection
        wordembs = prepare_dataset("edu.stanford.glove.6b.50")        
        vocab = WordvecUnkVocab(data=wordembs, random=random)
        robust = RobustDataset.prepare()

        # Train with OpenNIR DRMM model
        ranker = Drmm(vocab=vocab).tag("ranker", "drmm")
        predictor = Reranker()
        trainer = PointwiseTrainer()
        learner = Learner(trainer=trainer, random=random, ranker=ranker, valid_pred=predictor, 
            train_dataset=robust('trf1'), val_dataset=robust('vaf1'), max_epoch=max_epoch)
        model = learner.submit()

        # Evaluate
        Evaluate(dataset=robust('f1'), model=model, predictor=predictor).submit()


if __name__ == "__main__":
    cli()
```

Start with (using the folder drmm-test to store the ouputs)
```sh
python test.py --port 12345 drmm-test
```

## Features

The features below are from [OpenNIR](http://github.com/)
### Rankers

Available in the `onir.rankers` module

 - DRMM `onir.rankers.drmm.Drmm` [paper](https://arxiv.org/abs/1711.08611)
 - (*planned*) Duet (local model) [paper](https://arxiv.org/abs/1610.08136)
 - (*planned*) MatchPyramid  [paper](https://arxiv.org/abs/1606.04648)
 - (*planned*) KNRM [paper](https://arxiv.org/abs/1706.06613)
 - (*planned*) PACRR  [paper](https://arxiv.org/abs/1704.03940)
 - (*planned*) ConvKNRM  [paper](https://www.semanticscholar.org/paper/432b36c1bec275c2778c66f9897f9e02f7d8b579)
 - (*planned*) Vanilla BERT `config/vanilla_bert` [paper](https://arxiv.org/abs/1810.04805)
 - CEDR models `onir.rankers.cedr_drmm.CedrDrmm` [paper](https://arxiv.org/abs/1810.04805)
 - (*planned*) MatchZoo models [source](https://github.com/NTMC-Community/MatchZoo)
 - (*planned*) MatchZoo's KNRM 
 - (*planned*) MatchZoo's ConvKNRM 

### Datasets

 - TREC Robust 2004
 - (*planned*) MS-MARCO `config/msmarco`
 - (*planned*) ANTIQUE `config/antique`
 - (*planned*) TREC CAR `config/car`
 - (*planned*) New York Times `config/nyt` -- for [content-based weak supervision](https://arxiv.org/abs/1707.00189)
 - (*planned*) TREC Arabic, Mandarin, and Spanish `config/multiling/*` -- for [zero-shot multilingual transfer learning](https://arxiv.org/pdf/1912.13080.pdf) ([instructions](https://opennir.net/multilingual.html))

### Evaluation Metrics

 - `map` (from trec_eval)
 - `ndcg` (from trec_eval)
 - `ndcg@X` (from trec_eval, gdeval)
 - `p@X` (from trec_eval)
 - `err@X` (from gdeval)
 - `mrr` (from trec_eval)
 - `rprec` (from trec_eval)
 - `judged@X` (implemented in python)

### Vocabularies

 - (**planned**) Binary term matching `vocab=binary` (i.e., changes interaction matrix from cosine similarity to to binary indicators)
 - Pretrained word vectors. Find the list with `datamaestro search tag:"word embeddings"`

## Citing OpenNIR

If you use OpenNIR, please cite the real OpenNIR WSDM demonstration paper and
look at acknowledgements of the original [OpenNIR](https://github.com/Georgetown-IR-Lab/OpenNIR).

```
@InProceedings{macavaney:wsdm2020-onir,
  author = {MacAvaney, Sean},
  title = {{OpenNIR}: A Complete Neural Ad-Hoc Ranking Pipeline},
  booktitle = {{WSDM} 2020},
  year = {2020}
}
```

If you have space, you can also cite mine:

```bibtex
@inproceedings{10.1145/3397271.3401410,
author = {Piwowarski, Benjamin},
title = {Experimaestro and Datamaestro: Experiment and Dataset Managers (for IR)},
year = {2020},
doi = {10.1145/3397271.3401410},
booktitle = {Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
location = {Virtual Event, China},
series = {SIGIR ’20}
}
  

```
