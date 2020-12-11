python src/whitebox_eval.py --gpu 2 --target-model-type small --target-loss-type center --target-dataset-type vgg \
--model-type small --loss-type triplet --dataset-type vgg --attack PGD --norm 2 --targeted-flag true --tv-flag false \
--hinge-flag true --margin 15.0 --amplification 8.0 --granularity coarser --cos-flag false

python src/whitebox_eval.py --gpu 2 --target-model-type large --target-loss-type triplet --target-dataset-type vgg \
--model-type small --loss-type triplet --dataset-type vgg --attack PGD --norm 2 --targeted-flag true --tv-flag false \
--hinge-flag true --margin 15.0 --amplification 8.0 --granularity coarser --cos-flag false

python src/whitebox_eval.py --gpu 2 --target-model-type large --target-loss-type center --target-dataset-type vgg \
--model-type small --loss-type triplet --dataset-type vgg --attack PGD --norm 2 --targeted-flag true --tv-flag false \
--hinge-flag true --margin 15.0 --amplification 8.0 --granularity coarser --cos-flag false

python src/whitebox_eval.py --gpu 2 --target-model-type large --target-loss-type center --target-dataset-type casia \
--model-type small --loss-type triplet --dataset-type vgg --attack PGD --norm 2 --targeted-flag true --tv-flag false \
--hinge-flag true --margin 15.0 --amplification 8.0 --granularity coarser --cos-flag false
