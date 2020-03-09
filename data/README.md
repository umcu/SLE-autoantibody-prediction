# SLE datasets

## `/processed`

These datasets are ready for analysis

* `data\processed\20190717_Total_Norm-3.xlsx`: SLE dataset used for TB's papers

## `/interim`

These datasets were exported by TB or TF and are used to build the data frame for further analysis / machine learning.

* `data\processed\TMO.csv`: From TF, which is what they used to build their model. *Use this to get data from blood-bank controls*
    - **Rows**: 505 SLE patients, 361 blood bank donors
    - **Columns**: 74 features, 1 class column
* `data\interim\Alles 1e sample.sav`: From TB; data from first sample for all patients. *Use this to get TB's selection of all SLE patients* (484, cf. 505 from TF).
    * **Rows**: 2038 patients
    * **Columns**: 374 (including 74 features from `TMO.csv`)
* `data\interim\Non-Imid control set.sav`: From TB; data from patients who had 2 or less SLE-related symptoms and no symptoms suspect for other immune-mediated inflammatory disorder (IMID, e.g. arthritis, nephritis, pleuritis. There are actually 218 of these, but one has no chip-data from TF available.
    * **Rows**: 218 patients
    * **Columns**: 368 (including 74 features from `TMO.csv`)
* `data\interim\OtherIMID.sav`: From TB; data from patients who do not have SLE but other IMIDs (e.g. anti-phospholipid syndrome)
    * **Rows**: 346 patients
    * **Columns**: 368 (including 74 features from `TMO.csv`)

# Text mining

Both of these are used by `notebooks/text-mining-validation.Rmd`

* `data\processed\ds_reproduced.csv`: Output of text mining scripts when run with parameters from the paper
* `data\processed\ds_1d99dca.csv`: Output of adapted text mining scripts (after commit `1d99dca`: "Fix detection counters for diagnoses")