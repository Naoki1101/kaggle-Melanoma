cd ../src

# python train.py -m 'resnet18_001' -c 'test'
python train.py -m 'resnet18_002' -c 'base model'

cd ../
git add -A
git commit -m '...'
git push origin master