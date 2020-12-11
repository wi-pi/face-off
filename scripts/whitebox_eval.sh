python src/whitebox_eval.py --gpu 1 --target-model-type small --target-loss-type center --target-dataset-type vgg \
--model-type large --loss-type triplet --dataset-type vgg --attack CW --norm 2 --targeted-flag true --tv-flag false \
--hinge-flag true --margin 15.0 --amplification 8.0 --granularity coarser --cos-flag false --pair-flag false

python src/whitebox_eval.py --gpu 1 --target-model-type small --target-loss-type triplet --target-dataset-type vgg \
--model-type large --loss-type triplet --dataset-type vgg --attack CW --norm 2 --targeted-flag true --tv-flag false \
--hinge-flag true --margin 15.0 --amplification 8.0 --granularity coarser --cos-flag false --pair-flag false

python src/whitebox_eval.py --gpu 1 --target-model-type large --target-loss-type center --target-dataset-type vgg \
--model-type large --loss-type triplet --dataset-type vgg --attack CW --norm 2 --targeted-flag true --tv-flag false \
--hinge-flag true --margin 15.0 --amplification 8.0 --granularity coarser --cos-flag false --pair-flag false

python src/whitebox_eval.py --gpu 1 --target-model-type large --target-loss-type center --target-dataset-type casia \
--model-type large --loss-type triplet --dataset-type vgg --attack CW --norm 2 --targeted-flag true --tv-flag false \
--hinge-flag true --margin 15.0 --amplification 8.0 --granularity coarser --cos-flag false --pair-flag false
