# Face-Off

### Background
Face-Off is a privacy-preserving framework that introduces strategic perturbations to images of the user’s face to prevent it from being correctly recognized. By leveraging adversarial examples generated on faces in a black-box setting, we find that our perturbations transfer to proprietary face recognition APIs such as AWS Rekognition, Azure Face, and Face++.

### Installation
Using python version 3.5.2
```
python3.5 -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
pip install -e .
pip install -r requirements.txt
```

### Required weights
```
https://drive.google.com/drive/folders/1qE21bgqeCjtqyUrCqehRd8MKt4zGn-i4?usp=sharing
```

### Development
```
./scripts/create_directories.sh
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

### API Evaluation
Note that any API evaluation requires accounts, keys, and an AWS S3 bucket. Below are some links to resources helpful for setting up keys.
https://docs.aws.amazon.com/AmazonS3/latest/dev/access-points.html
https://aws.amazon.com/premiumsupport/knowledge-center/read-access-objects-s3-bucket/
https://docs.aws.amazon.com/rekognition/latest/dg/getting-started.html
https://azure.microsoft.com/en-us/services/cognitive-services/face/#get-started
https://www.faceplusplus.com/

## Link to paper
https://arxiv.org/abs/2003.08861
