python src/api_eval.py --api-name facepp --model-type large --loss-type triplet --dataset-type vgg --attack PGD \
--norm 2 --targeted-flag true --tv-flag false --hinge-flag true --margin 15.0 --amplification 10.6 --interpolation bilinear \
--granularity api-eval --topn true --pair-flag false --credentials 0 --mean-loss embedding

#python src/api_eval.py --api-name facepp --model-type small --loss-type center --dataset-type vgg --attack PGD \
#--norm 2 --targeted-flag true --tv-flag false --hinge-flag true --margin 15.0 --amplification 10.6 --interpolation bilinear \
#--granularity api-eval --topn true --pair-flag false --credentials 1 --mean-loss embedding
