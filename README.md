Multi-Task Argument Mining
==============================


This repository contains the code and results for my bachelor’s thesis:

- **Title:** Multi-Task Argument Mining
- **Author:** Johanna Reiml  
- **University:** Ludwig-Maximilians-Universität München
- **Supervised by:** Prof. Dr. Thomas Seidl  
- **Advised by:** Dr. Michael Fromm  
- **Submission Date:** February 10, 2022

Abstract
------------

In this project, we explore the relationship between various argument related tasks, and investigate whether multi-task learning can improve the performance when fine-tuning argument mining tasks on the pre-trained language model BERT. In particular, we argue that the domain of topic information plays an important role in argument mining, since it provides semantic and contextual information for an argument. For this, we explore various settings such as cross-topic and in-topic scenarios. To the best of our best knowledge, we are the first to compare AM task similarities by examining how individual tasks can benefit from each other, while also exploring the effect of topic information.

--------

Results
------------

In order to reproduce the results, we first need to populate the repository with the raw datasets. These can be obtained here.

- [IBM Debater® - Evidences Sentences](https://research.ibm.com/haifa/dept/vst/debating_data.shtml)
- [IBM Debater® - IBM-ArgQ-Rank-30kArgs](https://research.ibm.com/haifa/dept/vst/debating_data.shtml)
- [UKP Sentential Argument Mining Corpus](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2345)

Once downloaded, the datasets have to be placed in the following directories.

    ├── data
    │   └── raw
    │       └── arg_quality_rank_30k
    │           └── arg_quality_rank_30k.csv
    │       └── evidences_sentences
    │           └── wikipedia_evidence_dataset_29429.csv
    │       └── UKP_sentential_argument_mining
    │           └── data
    │               └── abortion.tsv
    │               └── cloning.tsv
    │               └── death_penalty.tsv
    │               └── gun_control.tsv
    │               └── marijuana_legalization.tsv
    │               └── minimum_wage.tsv
    │               └── nuclear_energy.tsv
    │               └── school_uniforms.tsv

Once all datasets are in place, results can be generated using the `make results` command, which will:
1) Pre-process all raw datasets.
2) Run training and prediction for all of our models on multiple seeds.
3) Generate final results across all seeds with mean/std.

The final results can be found in the `results` directory.

--------

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── results            <- Logged results from training and prediction tasks.
    │
    ├── scripts            <- Scripts for running model training and prediction.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to turn raw data into features for modeling
    │   │   └── make_dataset.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── results  <- Scripts to process results obtained by model trainind and predictions
    │       └── process_results.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

Citing
--------

If you use this work, please cite it as:

```bibtex
@mastersthesis{reiml2022argument,
  title={Multi-Task Argument Mining},
  author={Johanna Reiml},
  type={Bachelor's Thesis},
  school={Ludwig-Maximilians-Universität München},
  year={2022},
  note={\url{https://github.com/jreiml/bachelor-thesis-argument-mining}}
}
```

If you find this work relevant, I recommend our derivative work:

```bibtex
@misc{fromm2022holistic,
      title={Towards a Holistic View on Argument Quality Prediction}, 
      author={Michael Fromm and Max Berrendorf and Johanna Reiml and Isabelle Mayerhofer and Siddharth Bhargava and Evgeniy Faerman and Thomas Seidl},
      year={2022},
      eprint={2205.09803},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

--------

License
--------

This work is licensed under the CC BY-SA 4.0 License.


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
