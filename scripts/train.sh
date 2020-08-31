for i in 2 3 4 5 8 16 32;
do
  for j in 2 3 4 5 8 16 32;
  do
    for l in 2 4 8 16 32 64 128;
    do
      echo "CUDA_VISIBLE_DEVICES=9 python ../models/train.py $i-$j-0 --model lstm --cuda --output /raid/lingo/abau/random-walks/lstm-$i-$j-0-$l/ --max_length $l"
      CUDA_VISIBLE_DEVICES=9 python ../models/train.py $i-$j-0 --model lstm --cuda --output /raid/lingo/abau/random-walks/lstm-$i-$j-0-$l/ --max_length $l
    done
  done
done
