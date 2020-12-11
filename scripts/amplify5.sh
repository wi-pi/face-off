python src/amplify.py --gpu 2 --model-type small --loss-type center --dataset-type vgg --attack PGD --norm 2 --targeted-flag true \
--tv-flag false --hinge-flag true --margin 15.0 --amplification 10.6 --interpolation bilinear --granularity api-eval \
--cos-flag false --mean-loss embedding --pair-flag false
