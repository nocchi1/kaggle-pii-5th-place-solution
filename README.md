# The Learning Agency Lab - PII Data Detection - 5th Place Solution
![certificate](./appendix/certificate.png)
This repository contains the code for the 5th place solution (Ryota Part) in the "PII Data Detection" competition hosted on Kaggle.
In this competition, participants were given the task of developing machine learning models to detect personally identifiable information (PII) in student writing. The objective is to create automated methods for identifying and removing PII from educational datasets, thereby facilitating the release of these datasets for research and development of educational tools while ensuring student privacy.

## Solution Summary
My solution is training models based on [DeBERTa-V3-large](https://huggingface.co/microsoft/deberta-v3-large). In addition to text tokens, I added features that indicate positional information within the essays. During the initial phase of training, I included external data created by an LLM along with the data provided in the competition. After that, I trained using only the competition data, which I considered to be of high quality. Also, by utilizing online label smoothing and taking into account the similarity between incorrect labels, I was able to improve accuracy.

For our team's final submission, we prepared 13 models primarily based on DeBERTa-V3-large and used them for voting ensemble. Although the backbone models used were similar, team members trained them with different settings to enhance diversity. The most notable difference was in the max_length parameter. We believed that by changing the context length, the models could detect PII from various perspectives.

For more details about our solution, please refer to the URL provided in the Links section.

## Preparation
You can set up the environment and download the required data by running the following commands.

### Setup
The environment is set up using [rye](https://rye.astral.sh/)
```sh
. ./bin/setup.sh
```

### Download Dataset
```sh
. ./bin/download.sh
```

## Reproducing the Solution
[TBU]

## Links
- Competition website : [link](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/leaderboard)
- 5th place solution : [link](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497306)
- My team members : [takai380](https://www.kaggle.com/takai380), [min fuka](https://www.kaggle.com/minfuka)
