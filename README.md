# Easy-Train HF (Alpha version)

## Using  Easy-Train HF

### Basic logic
The code in `train_model.py` can be used to train any Hugging-Face-compatible model with a HF-compatible tokenizer on any HF-compatible training dataset and calculate loss on a different HF-compatible dataset. These can all be uploaded onto the HF model or dataset hub, and are all specified in the model `config` file, examples of which are provided in `model_configs/` (currently contains one example).

### How to run

To train a model, run:

```
python train_model.py -c <path/to/config/file> -s <seed>
```

### Example

The current example (`model_configs/pythia160m.yaml`) trains the `EleutherAI/pythia-160m` model on a subset of Fineweb-edu (streamed directly from the HF dataset hub) for 1000 steps and saves and evaluates this model on a different local subset of the same dataset (see `test_sets/create_test_set.py` for how this validation set was generated). Following [Michaelov et al. (2025)](#citation), the model has a context length of `1024` and is trained on effective batch size `512` (note that `batch_size` and `gradient accumulation` may need to be updated depending on your GPU setup).

To run the test example with seed `42`, run:

```
python train_model.py -c model_configs/pythia160m.yaml -s 42
```


## Citation
If using, please cite the paper in which the original version of this script was first introduced:

```
@inproceedings{michaelov_2025_neurips,
  title={Language Model Behavioral Phases are Consistent Across Architecture, Training Data, and Scale},
  author={Michaelov, James A. and Levy, Roger P. and Bergen, Benjamin K.},
  booktitle={The Thirty-Ninth Annual Conference on Neural Information Processing Systems}
}

```
