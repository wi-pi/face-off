python src/api_eval.py --api-name azure --model-type large --loss-type triplet --dataset-type vgg --attack CW \
--norm 2 --targeted-flag true --tv-flag false --hinge-flag true --margin 15.0 --amplification 10.6 --interpolation bilinear \
--granularity api-eval --topn true --pair-flag false --credentials 1 --mean-loss embedding

#python src/api_eval.py --api-name azure --model-type small --loss-type center --dataset-type vgg --attack CW \
#--norm 2 --targeted-flag true --tv-flag false --hinge-flag true --margin 15.0 --amplification 10.6 --interpolation bilinear \
#--granularity api-eval --topn true --pair-flag true --credentials 1 --mean-loss embedding
