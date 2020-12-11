python src/attack.py --gpu 2 --model-type large --loss-type center --dataset-type casia --attack CW \
--norm 2 --targeted-flag true --tv-flag false --hinge-flag true --epsilon 0.1 --margin 15.0 --amplification 10.6 \
--iterations 100 --binary-steps 8 --learning-rate 0.01 --epsilon-steps 0.01 --init-const 0.3 --interpolation bilinear \
--granularity api-eval --batch-size 0 --pair-flag true
