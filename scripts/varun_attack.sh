python src/attack.py --gpu 0 --model-type large --loss-type center --dataset-type casia --attack CW \
--norm inf --targeted-flag true --tv-flag false --hinge-flag true --epsilon 0.1 --margin 15.0 --amplification 6.0 \
--iterations 50 --binary-steps 2 --learning-rate 0.01 --epsilon-steps 0.01 --init-const 0.3 --interpolation bilinear \
--granularity coarser --batch-size 0
