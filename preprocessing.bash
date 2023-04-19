index_array=(0 96 192 288 384 480 576 672 768 864 960 1056 1152 1248 1344 1440 1535)

for i in {1..16}
do
   nohup python preprocessing.py ${index_array[i-1]} ${index_array[i]} ${i} > log/${i}.log 2>&1 &
done