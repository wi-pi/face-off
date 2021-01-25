# Face-Off

### Background
Face-Off is a privacy-preserving framework that introduces strategic perturbations to images of the user’s face to prevent it from being correctly recognized. By leveraging adversarial examples generated on faces in a black-box setting, we find that our perturbations transfer to proprietary face recognition APIs such as AWS Rekognition, Azure Face, and Face++.

### Installation
Using python version 3.5.2
```
git clone https://github.com/wi-pi/face-off.git
python -m pip install venv
python3.5 -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
pip install -e .
pip install -r requirements.txt
```

### Required weights
https://drive.google.com/drive/folders/1qE21bgqeCjtqyUrCqehRd8MKt4zGn-i4?usp=sharing
Download each of the weights to the  `./weights` folder. These are used for generating adversarial examples and evaluating transferability offline.

### Required resources
Using a server with 40 CPU cores, 2 Nvidia TITAN Xp's, and 1 Quadro P6000, 125 GB Memory, Ubuntu version 16.04 LTS.
Using CUDA 10.0, NVIDIA Driver 410.104.
See https://stackoverflow.com/questions/50622525/which-tensorflow-and-cuda-version-combinations-are-compatible for CUDA - TensorFlow compatibility.

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
./scripts/mask_my_face.sh
```
Generates perturbations on the set of faces in `data/myface/`
NOTE: If you want to use hinge loss, you must align a bucket of your own faces to sizes 160x160 or 96x96. You can use MTCNN to do so. We will integrate support for this shortly.
```
./scripts/api_eval.sh
```
Feeds generated perturbations into the APIs and stores results
```
./scripts/analyze_api_results.sh
```
Reads and interprets the API scores

### API evaluation
Note that any API evaluation requires accounts, keys, and an AWS S3 bucket. Below are some links to resources helpful for setting up keys. Follow the step-by-step instructions found in the below links.

AWS S3: https://docs.aws.amazon.com/AmazonS3/latest/dev/access-points.html

AWS S3 - Public Read Access: https://aws.amazon.com/premiumsupport/knowledge-center/read-access-objects-s3-bucket/

AWS Rekognition: https://docs.aws.amazon.com/rekognition/latest/dg/getting-started.html

Azure Face: https://azure.microsoft.com/en-us/services/cognitive-services/face/#get-started

Face++: https://console.faceplusplus.com/documents/7079083

### Link to the paper
https://arxiv.org/abs/2003.08861

### Citation
```
@inproceedings{faceoff-pets21,
  title = {Face-Off: Adversarial Face Obfuscation},
  author = {Varun Chandrasekaran, Chuhan Gao, Brian Tang, Kassem Fawaz, Somesh Jha, and Suman Banerjee},
  booktitle = {Privacy Enhancing Technologies 2021 (PETS 2021)},
  year = {2021}
}
```

### Contact
Questions? Contact bjtang2@wisc.edu or byron123t@gmail.com with subject: Face-Off
