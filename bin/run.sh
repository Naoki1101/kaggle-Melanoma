cd ../src

# python train.py -m 'efficientnetb3_001' -c 'size=384'
# python train.py -m 'efficientnetb3_002' -c 'ReduceLROnPlateau, lr=0.01'
# python train.py -m 'efficientnetb3_003' -c 'epoch=40'
# python train.py -m 'efficientnetb3_004' -c 'modify head'
# python train.py -m 'efficientnetb3_005' -c 'add one-hot features'
# python train.py -m 'efficientnetb3_006' -c 'add one-hot target features'
# python train.py -m 'efficientnetb3_007' -c 'only target features'
# python train.py -m 'efficientnetb3_008' -c 'add rotate, cutout, rrcrop'
# python train.py -m 'efficientnetb3_009' -c 'add microscope'
# python train.py -m 'efficientnetb3_010' -c 'StratifiedGroupKFold'
# python train.py -m 'efficientnetb3_011' -c 'StratifiedGroupKFold, 5fold'
# python train.py -m 'efficientnetb3_012' -c 'image only, StratifiedGroupKFold, 5fold'
# python train.py -m 'efficientnetb3_013' -c 'add external data'
# python train.py -m 'efficientnetb3_014' -c 'epoch=20, external data, target_encoding'
# python train.py -m 'efficientnetb3_015' -c 'epoch=20, external data'
# python train.py -m 'efficientnetb3_016' -c 'tta=5'
# python train.py -m 'efficientnetb3_017' -c '5fold, tta=5'
# python train.py -m 'efficientnetb3_018' -c 'WeightedFocalLoss'
# python train.py -m 'efficientnetb3_019' -c '5fold, tta=5, WeightedFocalLoss'
# python train.py -m 'efficientnetb3_020' -c '512, 5fold, tta=5, WeightedFocalLoss'
# python train.py -m 'efficientnetb3_021' -c '512, single fold, tta=10, WeightedFocalLoss'
# python train.py -m 'efficientnetb3_022' -c '512, single fold, tta=10, WeightedFocalLoss, remove Microscope'
python train.py -m 'efficientnetb3_022' -c '512, single fold, tta=10, WeightedFocalLoss, modify head'

# python train.py -m 'efficientnetb4_001' -c 'external data'
# python train.py -m 'efficientnetb4_002' -c '5fold, tta=5'

# python train.py -m 'resnet18_001' -c 'test'
# python train.py -m 'resnet18_002' -c 'base model'
# python train.py -m 'resnet18_003' -c 'modify head'
# python train.py -m 'resnet18_004' -c 'modify head'
# python train.py -m 'resnet18_005' -c 'modify head'
# python train.py -m 'resnet18_006' -c 'add rotate, cutout'

# python train.py -m 'se_resnext50_32x4d_001' -c 'base model'
# python train.py -m 'se_resnext50_32x4d_002' -c 'size=512'

# python ensemble.py -m 'ensemble_001' -c '3models'
# python ensemble.py -m 'ensemble_002' -c '3models, rank_average'
# python ensemble.py -m 'ensemble_003' -c '4models, rank_average'
# python ensemble.py -m 'ensemble_004' -c '5models, rank_average'
# python ensemble.py -m 'ensemble_005' -c '4models, rank_average'
# python ensemble.py -m 'ensemble_006' -c '5models, rank_average'

cd ../
git add -A
git commit -m '...'
git push origin master