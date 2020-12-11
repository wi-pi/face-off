python src/attack.py --gpu 2 --model-type large --loss-type triplet --dataset-type vgg --attack PGD \
--norm 2 --targeted-flag true --tv-flag false --hinge-flag true --epsilon 0.01 --margin 15.0 --amplification 8.0 \
--iterations 20 --binary-steps 8 --learning-rate 0.01 --epsilon-steps 0.01 --init-const 0.3 --interpolation bilinear \
--granularity coarser --batch-size 0
