#!/bin/sh

cd data
python ../src/kaggle/download.py -c pii-detection-removal-from-educational-data -p input -u

cd external

# dataset from https://www.kaggle.com/datasets/nbroad/pii-dd-mistral-generated
kaggle datasets download nbroad/pii-dd-mistral-generated -f mixtral-8x7b-v1.json
unzip mixtral-8x7b-v1.json.zip
rm mixtral-8x7b-v1.json.zip
mv mixtral-8x7b-v1.json nicholas.json

# dataset from https://www.kaggle.com/datasets/mpware/pii-mixtral8x7b-generated-essays
kaggle datasets download mpware/pii-mixtral8x7b-generated-essays -f mpware_mixtral8x7b_v1.1-no-i-username.json
unzip mpware_mixtral8x7b_v1.1-no-i-username.json.zip
rm mpware_mixtral8x7b_v1.1-no-i-username.json.zip
mv mpware_mixtral8x7b_v1.1-no-i-username.json mpware.json

# dataset from https://www.kaggle.com/datasets/pjmathematician/pii-detection-dataset-gpt
# revised based on https://www.kaggle.com/code/valentinwerner/fix-punctuation-tokenization-external-dataset
kaggle datasets download sujithkumarmanickam/pii-extra-data -f moredata_dataset_fixed.json
unzip moredata_dataset_fixed.json.zip
rm moredata_dataset_fixed.json.zip
mv moredata_dataset_fixed.json pjma.json