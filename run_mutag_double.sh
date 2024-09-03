dataset=MUTAG #NCI1 PROTEINS DD MUTAG COLLAB REDDIT-BINARY
A=gin
B=gcn
C=gat
D=set
python train.py --dataset $dataset --encoder_1 $A --encoder_2 $A &
python train.py --dataset $dataset --encoder_1 $A --encoder_2 $B &
python train.py --dataset $dataset --encoder_1 $A --encoder_2 $C &
python train.py --dataset $dataset --encoder_1 $A --encoder_2 $D
#python train.py --dataset $dataset --encoder_1 $B --encoder_2 $B &
#python train.py --dataset $dataset --encoder_1 $B --encoder_2 $C &
#python train.py --dataset $dataset --encoder_1 $B --encoder_2 $D &
#python train.py --dataset $dataset --encoder_1 $C --encoder_2 $C &
#python train.py --dataset $dataset --encoder_1 $C --encoder_2 $D
