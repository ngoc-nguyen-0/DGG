# python3 LCVAEv4.py --dataset mnist --resume "" --code_length 10 --batch_size 100 --num_epochs 250
# python3 LCVAEv11.py --dataset "reuters10k" --resume "" --code_length 4 --batch_size 100 --num_epochs 250
# python3 VaDE.py --dataset "reuters10k" --resume "" --code_length 4 --batch_size 100 --num_epochs 250
# python3 VaDE.py --dataset "mnist" --resume "" --code_length 10 --batch_size 100 --num_epochs 250
python3 DAE.py --dataset "reuters10k" --resume "" --code_length 4 --batch_size 100 --num_epochs 250
