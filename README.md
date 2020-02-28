# SLE

This project aims to improve the diagnosis of [Systemic Lupus Erythematosus](https://en.wikipedia.org/wiki/Systemic_lupus_erythematosus) using text mining and predictive modelling.

## Text mining

Clinical data (whether a person is considered to have SLE or not) is extracted from the electronic health records. 

See the (R) scripts in `scripts/text-mining` and the `text-mining-validation` notebook in `notebooks`.

## Predictive modelling

The next step is to use this data to predict SLE diagnosis using blood biomarkers such as auto-antibodies. 

See the (Python) code in `src/sle` and the `notebooks` folder.
