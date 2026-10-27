mode: full

## Bruteforce_CWE-307
- rule: mean (chosen on tuning half)
- train windows: 150000 (cap 150000)
- LSTM hidden 64, layers 1, L 20, max epochs 20
- model seeds: 3
```
[LID-DS Bruteforce_CWE-307: TEST half] mean +/- std over 3 model seeds
threshold              Prec          Recall        F1            FAR           AUC
calib99               0.684+/-0.008 1.0+/-0.0     0.812+/-0.006 0.084+/-0.003 1.0+/-0.0
calib95               0.631+/-0.017 1.0+/-0.0     0.774+/-0.013 0.107+/-0.008 1.0+/-0.0
tuned_on_tuning_half  1.0+/-0.0     1.0+/-0.0     1.0+/-0.0     0.0+/-0.0     1.0+/-0.0
ms/window (1 thread): 0.0338 | params: 28968 | size KB: 116.0
```
mode: full

## Bruteforce_CWE-307
- rule: mean (chosen on tuning half)
- train windows: 150000 (cap 150000)
- LSTM hidden 64, layers 1, L 20, max epochs 20
- model seeds: 3
```
[LID-DS Bruteforce_CWE-307: TEST half] mean +/- std over 3 model seeds
threshold              Prec          Recall        F1            FAR           AUC
calib99               0.684+/-0.008 1.0+/-0.0     0.812+/-0.006 0.084+/-0.003 1.0+/-0.0
calib95               0.631+/-0.017 1.0+/-0.0     0.774+/-0.013 0.107+/-0.008 1.0+/-0.0
tuned_on_tuning_half  1.0+/-0.0     1.0+/-0.0     1.0+/-0.0     0.0+/-0.0     1.0+/-0.0
ms/window (1 thread): 0.0342 | params: 28968 | size KB: 116.0
```
