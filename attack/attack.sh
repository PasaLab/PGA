
#rate=(0.01 0.02 0.03 0.04 0.05)
#attacks=(pgdattack pgdattack-CW greedy)
#datasets=(cora citeseer cora_ml pubmed)
#
#for dataset in "${datasets[@]}"
#do
#  for att in "${attacks[@]}"
#  do
#    for ri in "${rate[@]}"
#    do
#      python -u gen_attack.py --save=True --gpu_id=1 --logger_level=1 --victim=gcn \
#        --ptb_rate=$ri \
#        --attack=$att \
#        --dataset=$dataset
#    done
#  done
#done



##### cora
# 0.01
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=cora --victim=gcn --attack=pgdattack --ptb_rate=0.01
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=cora --victim=gcn --attack=greedy --ptb_rate=0.01

# 0.02
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=cora --victim=gcn --attack=pgdattack --ptb_rate=0.02
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=cora --victim=gcn --attack=greedy --ptb_rate=0.02

# 0.03
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=cora --victim=gcn --attack=pgdattack --ptb_rate=0.03
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=cora --victim=gcn --attack=greedy --ptb_rate=0.03

# 0.04
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=cora --victim=gcn --attack=pgdattack --ptb_rate=0.04
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=cora --victim=gcn --attack=greedy --ptb_rate=0.04

# 0.05
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=cora --victim=gcn --attack=pgdattack --ptb_rate=0.05
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=cora --victim=gcn --attack=greedy --ptb_rate=0.05


##### citeseer
# 0.01
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=citeseer --victim=gcn --attack=pgdattack --ptb_rate=0.01
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=citeseer --victim=gcn --attack=greedy --ptb_rate=0.01

# 0.02
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=citeseer --victim=gcn --attack=pgdattack --ptb_rate=0.02
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=citeseer --victim=gcn --attack=greedy --ptb_rate=0.02

# 0.03
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=citeseer --victim=gcn --attack=pgdattack --ptb_rate=0.03
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=citeseer --victim=gcn --attack=greedy --ptb_rate=0.03

# 0.04
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=citeseer --victim=gcn --attack=pgdattack --ptb_rate=0.04
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=citeseer --victim=gcn --attack=greedy --ptb_rate=0.04

# 0.05
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=citeseer --victim=gcn --attack=pgdattack --ptb_rate=0.05
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=citeseer --victim=gcn --attack=greedy --ptb_rate=0.05


##### cora_ml
# 0.01
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=cora_ml --victim=gcn --attack=pgdattack --ptb_rate=0.01
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=cora_ml --victim=gcn --attack=greedy --ptb_rate=0.01

# 0.02
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=cora_ml --victim=gcn --attack=pgdattack --ptb_rate=0.02
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=cora_ml --victim=gcn --attack=greedy --ptb_rate=0.02

# 0.03
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=cora_ml --victim=gcn --attack=pgdattack --ptb_rate=0.03
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=cora_ml --victim=gcn --attack=greedy --ptb_rate=0.03

# 0.04
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=cora_ml --victim=gcn --attack=pgdattack --ptb_rate=0.04
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=cora_ml --victim=gcn --attack=greedy --ptb_rate=0.04

# 0.05
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=cora_ml --victim=gcn --attack=pgdattack --ptb_rate=0.05
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=cora_ml --victim=gcn --attack=greedy --ptb_rate=0.05



#### pubmed
 0.01
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=pubmed --victim=gcn --attack=pgdattack --ptb_rate=0.01
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=pubmed --victim=gcn --attack=greedy --ptb_rate=0.01

 0.02
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=pubmed --victim=gcn --attack=pgdattack --ptb_rate=0.02
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=pubmed --victim=gcn --attack=greedy --ptb_rate=0.02

 0.03
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=pubmed --victim=gcn --attack=pgdattack --ptb_rate=0.03
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=pubmed --victim=gcn --attack=greedy --ptb_rate=0.03

 0.04
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=pubmed --victim=gcn --attack=pgdattack --ptb_rate=0.04
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=pubmed --victim=gcn --attack=greedy --ptb_rate=0.04

 0.05
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=pubmed --victim=gcn --attack=pgdattack --ptb_rate=0.05
python -u gen_attack.py --save=True --gpu_id=2 --logger_level=1 --dataset=pubmed --victim=gcn --attack=greedy --ptb_rate=0.05
