# Easy-Train HF (Alpha version)

To train a model, run:

```
python train_model.py -c <path/to/config/file> -s <seed>
```

For example, to run the test example with seed `42`, run:

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