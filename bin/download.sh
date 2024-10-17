#!/bin/sh

. .venv/bin/activate

kaggle competitions download -c pii-detection-removal-from-educational-data -p data/input

# ref : https://www.kaggle.com/datasets/nbroad/pii-dd-mistral-generated
kaggle datasets download nbroad/pii-dd-mistral-generated -f mixtral-8x7b-v1.json -p data/input/external

# ref : https://www.kaggle.com/datasets/mpware/pii-mixtral8x7b-generated-essays
kaggle datasets download mpware/pii-mixtral8x7b-generated-essays -f mpware_mixtral8x7b_v1.1-no-i-username.json -p data/input/external

# ref : https://www.kaggle.com/datasets/pjmathematician/pii-detection-dataset-gpt
# ref : https://www.kaggle.com/code/valentinwerner/fix-punctuation-tokenization-external-dataset
kaggle datasets download sujithkumarmanickam/pii-extra-data -f moredata_dataset_fixed.json -p data/input/external

if ! command -v unzip &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y unzip
fi

unzip -o data/input/pii-detection-removal-from-educational-data.zip -d data/input &> /dev/null
rm data/input/pii-detection-removal-from-educational-data.zip
for file in data/input/external/*.zip; do
    if [ -f "$file" ]; then
        unzip -o "$file" -d data/input/external &> /dev/null
        rm "$file"
    fi
done

cd data/input/external
mv mixtral-8x7b-v1.json nicholas.json
mv mpware_mixtral8x7b_v1.1-no-i-username.json mpware.json
mv moredata_dataset_fixed.json pjma.json
cd ../../..
