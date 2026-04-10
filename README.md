# YOCO++

The code base for the paper "YOCO++: Enhancing YOCO with KV Residual Connections for Efficient LLM Inference".

## Installation

To install the dependencies, you may run the following commands:

```sh
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Usage

Our implementation is based on HuggingFace `transformers`. We register a new model `lckv-llama`.

### Training

We use the same [training script](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py) as the original `transformers` library.
To train a model with 1.1B parameters on the `wikitext-103` dataset, You may run the following command:

```sh
bash run_clm.sh
```

### Inference

We use the same [inference script](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-generation/run_generation.py) as the original `transformers` library.
To perform inference, you may run the following command:

```sh
bash run_generation.sh
```

### Evaluation

We use [LM-Harness](https://github.com/EleutherAI/lm-evaluation-harness) to evaluate the model.
To evaluate the model, you may run the following command:

```sh
python test_harness.py --model_name_or_path ...
```

### Latency Testing

To test the latency of the model, you may run the following command:

```sh
python test_latency.py
```

## Acknowledgment

The repository is built upon the [LCKV](https://github.com/whyNLP/LCKV) repository.
