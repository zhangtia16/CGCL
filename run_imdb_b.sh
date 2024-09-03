dataset=IMDB-BINARY #NCI1 PROTEINS DD MUTAG COLLAB REDDIT-BINARY
A=gin
B=gcn
C=gat
D=set
python train_three_encoders.py --dataset $dataset --encoder_1 $A --encoder_2 $A --encoder_3 $A --node_attr 'onehot' &
python train_three_encoders.py --dataset $dataset --encoder_1 $A --encoder_2 $A --encoder_3 $B --node_attr 'onehot' &
python train_three_encoders.py --dataset $dataset --encoder_1 $A --encoder_2 $A --encoder_3 $C --node_attr 'onehot' &
python train_three_encoders.py --dataset $dataset --encoder_1 $A --encoder_2 $A --encoder_3 $D --node_attr 'onehot' &
python train_three_encoders.py --dataset $dataset --encoder_1 $A --encoder_2 $B --encoder_3 $B --node_attr 'onehot' &
python train_three_encoders.py --dataset $dataset --encoder_1 $A --encoder_2 $B --encoder_3 $C --node_attr 'onehot' &
python train_three_encoders.py --dataset $dataset --encoder_1 $A --encoder_2 $B --encoder_3 $D --node_attr 'onehot' &
python train_three_encoders.py --dataset $dataset --encoder_1 $A --encoder_2 $C --encoder_3 $C --node_attr 'onehot' &
python train_three_encoders.py --dataset $dataset --encoder_1 $A --encoder_2 $C --encoder_3 $D --node_attr 'onehot' &
python train_three_encoders.py --dataset $dataset --encoder_1 $A --encoder_2 $D --encoder_3 $D --node_attr 'onehot' &
python train.py --dataset $dataset --encoder_1 $A --encoder_2 $A --node_attr 'onehot' &
python train.py --dataset $dataset --encoder_1 $A --encoder_2 $B --node_attr 'onehot' &
python train.py --dataset $dataset --encoder_1 $A --encoder_2 $C --node_attr 'onehot' &
python train.py --dataset $dataset --encoder_1 $A --encoder_2 $D --node_attr 'onehot'