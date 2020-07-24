cd ../src

# python train.py -m 'resnet18_001' -c 'test'
# python train.py -m 'resnet18_002' -c 'base model'
# python train.py -m 'resnet18_003' -c 'modify head'
python train.py -m 'resnet18_004' -c 'modify head'

# python train.py -m 'se_resnext50_32x4d_001' -c 'base model'

cd ../
git add -A
git commit -m '...'
git push origin master