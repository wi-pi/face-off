python src/amplify.py --gpu 1 --model-type small --loss-type center --dataset-type vgg --attack CW --norm inf --targeted-flag true \
--tv-flag false --hinge-flag true --margin 15.0 --amplification 10.6 --interpolation bilinear --granularity api-eval \
--cos-flag false --mean-loss embedding --pair-flag false
