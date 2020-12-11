# Face-Off

### Background
Face-Off

### Installation
```
Using python version 3.5.2
python3.5 -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
pip install -e .
pip install -r requirements.txt
```

### Required files
```
cp -r github/AdvFace/adv_imgs ./YOUR_WORKING_DIR
cp -r github/AdvFace/api_results ./YOUR_WORKING_DIR
cp -r github/AdvFace/test_imgs ./YOUR_WORKING_DIR
cp -r github/AdvFace/aligned_imgs ./YOUR_WORKING_DIR
```

### Development
```
./create_directories.sh
```
Creates the necessary subdirectories
```
./scripts/attack.sh
```
Generates perturbations on a single image-class pair
```
./scripts/api_eval.sh
```
Feeds generated perturbations into the APIs and stores results
```
./scripts/analyze_api_results.sh
```
Reads and interprets the API scores

## Link to paper
https://arxiv.org/abs/2003.08861
