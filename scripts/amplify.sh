python src/amplify.py --gpu 0 --model-type large --loss-type triplet --dataset-type vgg --attack CW --norm inf --targeted-flag true \
--tv-flag false --hinge-flag true --margin 15.0 --amplification 5.4 --interpolation bilinear --granularity api-eval \
--cos-flag false --mean-loss embedding
