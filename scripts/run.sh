for seed in `seq 0 9`
do
    python src/train.py --device cuda:0 --seed $seed --exp_str optim_$seed --model NAE --data housing
done

for seed in `seq 0 9`
do
    python src/train.py --device cuda:0 --seed $seed --exp_str optim_$seed --model NAE_D --data housing
done

for seed in `seq 0 9`
do
    python src/train.py --device cuda:0 --seed $seed --exp_str optim_$seed --model NAE_E --data housing
done
