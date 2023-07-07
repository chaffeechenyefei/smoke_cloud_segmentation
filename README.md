# 烟雾分割
https://ushare.ucloudadmin.com/pages/viewpage.action?pageId=116916370

# 使用说明

## 训练
```commandline
nohup python -u train_unetr18bn_smoke100k.py --batchsize 128  --lr 0.01 --mode 0 >x2smoke100k.out 2>&1 &
nohup python -u train_unetr18bn_smoke100k.py --batchsize 128 --loss focalloss --mode 0 >x2smoke100k_focal.out 2>&1 &
```

## 测试
```commandline
python eval_unetr18bn_smoke100k.py --mode 0
python test_unetr18bn.py --mode 0 --datapath --result  ...
```
