dataset=NCI1 #NCI1 PROTEINS DD MUTAG COLLAB REDDIT-BINARY
A=gin
B=gcn
C=gat
D=set
python train_three_encoders.py --dataset $dataset --encoder_1 $A --encoder_2 $A --encoder_3 $A &
python train_three_encoders.py --dataset $dataset --encoder_1 $A --encoder_2 $A --encoder_3 $B &
python train_three_encoders.py --dataset $dataset --encoder_1 $A --encoder_2 $A --encoder_3 $C &
python train_three_encoders.py --dataset $dataset --encoder_1 $A --encoder_2 $A --encoder_3 $D &
python train_three_encoders.py --dataset $dataset --encoder_1 $A --encoder_2 $B --encoder_3 $B &
python train_three_encoders.py --dataset $dataset --encoder_1 $A --encoder_2 $B --encoder_3 $C &
python train_three_encoders.py --dataset $dataset --encoder_1 $A --encoder_2 $B --encoder_3 $D &
python train_three_encoders.py --dataset $dataset --encoder_1 $A --encoder_2 $C --encoder_3 $C &
python train_three_encoders.py --dataset $dataset --encoder_1 $A --encoder_2 $C --encoder_3 $D &
python train_three_encoders.py --dataset $dataset --encoder_1 $A --encoder_2 $D --encoder_3 $D &
python train.py --dataset $dataset --encoder_1 $A --encoder_2 $A &
python train.py --dataset $dataset --encoder_1 $A --encoder_2 $B &
python train.py --dataset $dataset --encoder_1 $A --encoder_2 $C &
python train.py --dataset $dataset --encoder_1 $A --encoder_2 $D
