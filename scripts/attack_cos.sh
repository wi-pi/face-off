python src/attack.py --gpu 2,1,0 --model-type large --loss-type triplet --dataset-type vgg --attack CW \
--norm 2 --targeted-flag true --tv-flag false --hinge-flag true --epsilon 0.1 --margin 15.0 --amplification 8.0 \
--iterations 800 --binary-steps 15 --learning-rate 0.01 --epsilon-steps 0.01 --init-const 0.3 --interpolation bilinear \
--granularity fine-tuned --cos-flag true