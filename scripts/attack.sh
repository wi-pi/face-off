sources=('n000636' 'n001370' 'n001513' 'n002140' 'n002537' 'n004534' 'n005374' 'n005789' 'n006421' 'n007862' 'n001000' 'n001374' 'n001632' 'n002222' 'n003222' 'n005081' 'n005674' 'n005877' 'n007092' 'n007892' 'n001100' 'n001421' 'n001638' 'n002513' 'n003242' 'n005089' 'n005677' 'n006220' 'n007562' 'n008638' 'n001270' 'n001431' 'n001892' 'n002531' 'n003542' 'n005140' 'n005760' 'n006221' 'n007634' 'n009000' 'n001292' 'n001433' 'n002100' 'n002533' 'n003562' 'n005160' 'n005780' 'n006270' 'n007638' 'n009270')
targets=('n009000' 'n001292' 'n001433' 'n002100' 'n002533' 'n003562' 'n005160' 'n005780' 'n006270' 'n007638' 'n009270' 'n001892' 'n002531' 'n003542' 'n005140' 'n005760' 'n006221' 'n007634' 'n003242' 'n005089' 'n005677' 'n006220' 'n007562' 'n008638' 'n001270' 'n001431' 'n001374' 'n001632' 'n002222' 'n003222' 'n005081' 'n005674' 'n005877' 'n007092' 'n007892' 'n001100' 'n001421' 'n001638' 'n002513' 'n000636' 'n001370' 'n001513' 'n002140' 'n002537' 'n004534' 'n005374' 'n005789' 'n006421' 'n007862' 'n001000')

for i in "${!sources[@]}"
do
	:
	if [ $t != $s ]; then
		python src/attack.py --gpu 2 --model-type large --loss-type triplet --attack CW --base-imgs ${sources[$i]} --src ${sources[$i]} --target ${targets[$i]} \
		--norm 2 --attack-type true --tv-flag false --hinge-flag true --epsilon 0.1 --margin 5.0 --amplification 2.0 \
		--iterations 100 --binary-steps 5 --learning-rate 0.01 --epsilon-steps 0.01 --init-const 0.3 --interpolation bilinear \
		--granularity single --cos-flag false --vgg-flag true --batch-size 0
	fi
done