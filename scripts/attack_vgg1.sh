python src/attack.py --gpu 2 --model-type large --loss-type center --dataset-type casia --attack PGD \
--norm 2 --targeted-flag true --tv-flag false --hinge-flag true --epsilon 0.1 --margin 5.0 --amplification 3.0 \
--iterations 60 --binary-steps 6 --learning-rate 0.01 --epsilon-steps 0.01 --init-const 0.3 --interpolation bilinear \
--granularity single --batch-size 30
