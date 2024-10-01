# LitCab
Code and benchmark for [LitCab: Lightweight Calibration of Language Models on Outputs of Varied Lengths](https://arxiv.org/abs/2310.19208)

## CaT
The directories for the CaT datasets are listed below:
| Dataset   | Directory     |
|--------|----------|
|  NQ  |   NQ   |
|  SciQ  |  sciq  |
|  TriviaQA  |   triviaqa   |
|  TruthfulQA  |   truthfulqa   |
|  WikiQA  |   wikiqa   |
|  BioGen  |   name_bio   |
|  WikiGen  |   factuality_prompt   |

The training and evaluation files for each dataset within the corresponding directory are:
| Dataset   | Train  |  Test  |
|--------|----------|----------|
|  NQ  |  train.txt    |  test.txt   |
|  SciQ  |  train.txt  |   test.txt  |
|  Triviaqa  |  train.txt  |   test.txt  |
|  TruthfulQA  |  train.txt    |  test.txt   |
|  WikiQA  |  train.txt    |  test.txt   |
|  BioGen  |  unlabeled_prompt_entities.txt    |   prompt_entities.txt  |
|  WikiGen  |  train.jsonl    |  test.jsonl   |

## Evaluating LLMs on CaT
To evaluate a language model for all phrase- and sentence-level datasets, run the following command:
```bash
cd script
bash get_baselines.sh <model>
```
where `<model>` is the name of the model. The script will download the model and evaluate it on all datasets. The results will be saved in the `script/log` directory.

Please note that we call the OpenAI GPT-4 api throught Azure for evaluation. Please set the environment variable `AZURE_OPENAI_KEY` to your OpenAI API key. Your can also mannually set the key in `src/get_gpt_correctness.py` Line 13.

Before evaluating models on Long-form Generation, please run the following command to download the WikiPedia corpus:
```bash
cd FActScore
python -m factscore.download_data
```
To evaluate a language model for BioGen, run the following command:
```bash
cd script
bash get_baseline_long.sh
```
where the names of LLMs are set in `script/get_baseline_long.sh` Line 3. The results will be saved in the `script/log` directory.

To evaluate a language model for WikiGen, run the following command:
```bash
cd script
bash get_baseline_fp.sh
```
where the names of LLMs are set in `script/get_baseline_fp.sh` Line 3. The results will be saved in the `script/log` directory.
