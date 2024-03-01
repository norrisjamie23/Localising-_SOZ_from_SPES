# Localising the Seizure Onset Zone from Single-Pulse Electrical Stimulation Responses with a Transformer

This project is used to train models to classify SPES responses as within/outside of the SOZ. For this, you will need to download the dataset from [here](https://openneuro.org/datasets/ds004080/versions/1.2.4). This is only required for patients with SOZ labels.

To train these models:

1. Run `pip install -r requirements.txt` to install the necessary dependencies
2. Use `create_dataset.ipynb` to create the relevant data files.
3. Train and evaluate models with `train_and_evaluate.ipynb`.
