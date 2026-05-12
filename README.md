# MARFMDA

MARFMDA is a multi-agent LLM-assisted framework for miRNA-disease association prediction.

## Environment

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```text
API_KEY=your_api_key
BASE_URL=your_openai_compatible_base_url
MODEL_NAME=your_model_name
```

## Dataset

The processed dataset files are stored under `dataset/`. Large binary files are tracked with Git LFS.

### Processed Data Files

| File name | Shape / type | Description |
| :--- | :--- | :--- |
| `miRNA_name.npy` | `(1245,)` | All miRNA names |
| `disease_name.npy` | `(2077,)` | All disease names |
| `lncRNA_name.npy` | `(557,)` | All lncRNA names |
| `miRNA_miRNA.npy` | `(1245, 1245)` | Sequence similarity between miRNAs |
| `disease_disease.npy` | `(2077, 2077)` | Semantic similarity between diseases |
| `lncRNA_lncRNA.npy` | `(557, 557)` | Sequence similarity between lncRNAs |
| `miRNA_disease.npy` | `(1245, 2077)` | Association between miRNAs and diseases |
| `miRNA_lncRNA.npy` | `(1245, 557)` | Association between miRNAs and lncRNAs |
| `disease_lncRNA.npy` | `(2077, 557)` | Association between diseases and lncRNAs |
| `common_set.pkl` | `dict` | Data matrices that do not change with dataset splitting |
| `test_set.pkl` | `dict` | Test set |
| `train_set.pkl` | `dict` | Train set |
| `test_set_boosted.json` | `json` | Default input for the main experiment |
| `test_set_1000.json` | `json` | Subset for ablation, hyperparameter, and strategy experiments |

### Raw Data Sources

| Raw file | Source | Used to generate |
| :--- | :--- | :--- |
| `alldata_v4.xlsx` | [HMDD v4.0](http://www.cuilab.cn/hmdd), miRNA-disease association data, version 2023.07 | `miRNA_name.npy`, `disease_name.npy`, `miRNA_disease.npy` |
| `desc2024.xml` | [MeSH](https://www.nlm.nih.gov/mesh/meshhome.html), Medical Subject Headings thesaurus, version 2024 | `disease_disease.npy` |
| `miRNA.dat` | [miRBase v22](https://mirbase.org/), microRNA sequences and annotations, version 2019 | `miRNA_miRNA.npy` |
| `lncrna-diseases_experiment.txt` | [lncRNASNP2](https://guolab.wchscu.cn/lncRNASNP/#!/), functional SNPs and mutations in lncRNAs, version 2018 | `disease_lncRNA.npy` |
| `mirnas_lncrnas_validated.txt` | [lncRNASNP2](https://guolab.wchscu.cn/lncRNASNP/#!/) | `miRNA_lncRNA.npy` |
| `outLncRNA.fa` | [NONCODE v6.0](http://www.noncode.org/index.php), ncRNA and lncRNA knowledge database, version 2021 | `lncRNA_lncRNA.npy` |

### Data Processing

The raw data preprocessing follows these steps:

1. Load raw data files.
2. Remove noisy symbols and aliases, including special symbols such as `\xa0` in HMDD v4.0.
3. Align names within each database.
4. Merge miRNA, disease, and lncRNA data.
5. Align entity names across databases.
6. Match miRNA, disease, and lncRNA entities using the miRNA-miRNA, disease-disease, and lncRNA-lncRNA databases.
7. Screen miRNA-disease, miRNA-lncRNA, and disease-lncRNA associations by matched entities.
8. Calculate disease semantic similarity through directed acyclic graphs, following the method described in [Wang et al., Bioinformatics, 2010](https://doi.org/10.1093/bioinformatics/btq145).
9. Calculate miRNA sequence similarity from miRNA sequences.
10. Calculate lncRNA sequence similarity from lncRNA sequences.

Raw data preprocessing environment:

```text
python 3.9.0
pandas 2.0.1
numpy 1.23.5
biopython 1.83
```

### Training Preprocessing

`common_set.pkl` contains shared matrices:

| Key | Type | Shape | Description |
| :--- | :--- | :--- | :--- |
| `md` | `tensor.int64` | `(1245, 2077)` | Uncovered miRNA-disease association matrix, sum=23337 |
| `ml` | `tensor.int64` | `(1245, 557)` | Uncovered miRNA-lncRNA interaction matrix, sum=1438 |
| `dl` | `tensor.int64` | `(2077, 557)` | Uncovered disease-lncRNA association matrix, sum=320 |
| `mm_seq` | `tensor.float32` | `(1245, 1245)` | miRNA sequence similarity matrix |
| `dd_sem` | `tensor.float32` | `(2077, 2077)` | disease semantic similarity matrix |
| `ll_seq` | `tensor.float32` | `(557, 557)` | lncRNA sequence similarity matrix |
| `mm_mlG` | `tensor.float32` | `(1245, 1245)` | miRNA GIP kernel similarity from miRNA-lncRNA interactions |
| `dd_dlG` | `tensor.float32` | `(2077, 2077)` | disease GIP kernel similarity from disease-lncRNA associations |
| `ll_lmG` | `tensor.float32` | `(557, 557)` | lncRNA GIP kernel similarity from lncRNA-miRNA interactions |
| `ll_ldG` | `tensor.float32` | `(557, 557)` | lncRNA GIP kernel similarity from lncRNA-disease associations |
| `mm_mlF` | `tensor.float32` | `(1245, 1245)` | miRNA functional similarity from miRNA-lncRNA interactions |
| `dd_dlF` | `tensor.float32` | `(2077, 2077)` | disease functional similarity from disease-lncRNA associations |
| `ll_ldF` | `tensor.float32` | `(557, 557)` | lncRNA functional similarity from lncRNA-disease associations |
| `ll_lmF` | `tensor.float32` | `(557, 557)` | lncRNA functional similarity from lncRNA-miRNA interactions |

Training preprocessing uses k-fold cross validation, train/test splitting, and random seed control. Its environment is:

```text
python 3.10.12
numpy 1.25.0
scikit-learn 1.3.0
```

## File Description

| File_name | Description |
| :--- | :--- |
| `scripts/batch_predict_hybrid.py` | Main experimental pipeline |
| `agents/` | Multi-agent modules for RWR, similarity, graph features, fusion, and review |
| `data_loader/loader.py` | Dataset loading and preprocessing interface |
| `utils/` | Utility functions for scoring, graph processing, and token counting |
| `evaluation/evaluate.py` | Evaluation script |
| `ablations/` | Ablation study scripts |
| `config.py` | LLM and environment configuration |
| `requirements.txt` | Python dependencies |

## Run

Run the main experiment:

```bash
python scripts/batch_predict_hybrid.py
```

The default input file is:

```text
dataset/test_set_boosted.json
```

Evaluate prediction results:

```bash
python evaluation/evaluate.py --results results/*.json
```


