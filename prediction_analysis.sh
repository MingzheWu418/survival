#for i in {1..200}
#do
#  python evaluation_script.py --cancer_type "BLCA" --max_iter "$i"
#done

m="MLP"
a="KIPAN"
#python evaluation_script.py --cancer_type "$a" --max_iter 1
python evaluation_script.py --model_name "$m" --cancer_type "$a" --max_iter 10
#a="KICH"
#python evaluation_script.py --cancer_type "$a" --max_iter 50
python evaluation_script.py --model_name "$m" --cancer_type "$a" --max_iter 100
#a="KIPAN"
python evaluation_script.py --model_name "$m" --cancer_type "$a" --max_iter 500
#a="KIRC"
python evaluation_script.py --model_name "$m" --cancer_type "$a" --max_iter 1000
#a="LGG"
python evaluation_script.py --model_name "$m" --cancer_type "$a" --max_iter 2000