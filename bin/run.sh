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
# python train.py -m 'efficientnetb3_023' -c '512, single fold, tta=10, WeightedFocalLoss, modify head'
# python train.py -m 'efficientnetb3_024' -c 'mixup'
# python train.py -m 'efficientnetb3_025' -c '512, 5fold, tta=10, WeightedFocalLoss, modify head'
# python train.py -m 'efficientnetb3_026' -c 'tta=15'
# python train.py -m 'efficientnetb3_027' -c '512, 5fold, tta=10, WeightedFocalLoss, modify head, add age'
# python train.py -m 'efficientnetb3_028' -c '384, 1fold, tta=10, CosineAnnealingLR, brightness'
# python train.py -m 'efficientnetb3_029' -c '384, 1fold, tta=10, CosineAnnealingLR, brightness, max-avg'
# python train.py -m 'efficientnetb3_030' -c '384, 1fold, tta=10, hairAug'
# python train.py -m 'efficientnetb3_031' -c '384, 1fold, tta=10, hairRemove'
# python train.py -m 'efficientnetb3_032' -c '384, 1fold, tta=1, remove cutout'
# python train.py -m 'efficientnetb3_033' -c '384, 1fold, tta=1, remove cutout, lr=1.5e-5'
# python train.py -m 'efficientnetb3_034' -c '384, 1fold, tta=10, Augmix'
# python train.py -m 'efficientnetb3_035' -c '512, 1fold, tta=1, remove cutout'
# python train.py -m 'efficientnetb3_036' -c '384, 1fold, tta=1, Augmix'
# python train.py -m 'efficientnetb3_037' -c '384, 5fold, tta=10, Augmix'
# python train.py -m 'efficientnetb3_038' -c '384, 1fold, tta=1, Augmix, target_encoding_size'
# python train.py -m 'efficientnetb3_039' -c '384, 1fold, tta=1, Augmix, rondom crop'
# python train.py -m 'efficientnetb3_040' -c '384, 1fold, tta=1, Augmix, modify target_encoding_size'
# python train.py -m 'efficientnetb3_041' -c '256, 1fold, tta=1, Augmix'
# python train.py -m 'efficientnetb3_042' -c '384, 1fold, tta=1, Augmix, lr=4e-5'
# python train.py -m 'efficientnetb3_043' -c '384, 1fold, tta=1, Augmix, epoch=20'
# python train.py -m 'efficientnetb3_044' -c '384, 1fold, tta=1, modify augmentation, epoch=10'
# python train.py -m 'efficientnetb3_045' -c '256, 5fold, tta=12, Augmix, epoch=12, seed=2020'
# python train.py -m 'efficientnetb3_046' -c '384, 5fold, tta=10, modify augmentation, epoch=12, no metadata, seed=2020'
# python train.py -m 'efficientnetb3_047' -c '384, 5fold, tta=10, epoch=10, no metadata, seed=2021, drop_idx'
# python train.py -m 'efficientnetb3_048' -c '384, 5fold, tta=10, epoch=10, no metadata, seed=42, remove BN'

# python train.py -m 'efficientnetb4_001' -c 'external data'
# python train.py -m 'efficientnetb4_002' -c '5fold, tta=5'
# python train.py -m 'efficientnetb4_003' -c '512, 5fold, tta=10, WeightedFocalLoss, modify head'
# python train.py -m 'efficientnetb4_004' -c '384, 1fold, tta=1, Augmix, target_encoding_size'

# python train.py -m 'efficientnetb6_001' -c '256, 1fold, tta=1, Augmix'

# python train.py -m 'resnet18_001' -c 'test'
# python train.py -m 'resnet18_002' -c 'base model'
# python train.py -m 'resnet18_003' -c 'modify head'
# python train.py -m 'resnet18_004' -c 'modify head'
# python train.py -m 'resnet18_005' -c 'modify head'
# python train.py -m 'resnet18_006' -c 'add rotate, cutout'

# python train.py -m 'se_resnext50_32x4d_001' -c 'base model'
# python train.py -m 'se_resnext50_32x4d_002' -c 'size=512'
# python train.py -m 'se_resnext50_32x4d_003' -c '384, 1fold, tta=1, Augmix'
# python train.py -m 'se_resnext50_32x4d_004' -c '384, 5fold, tta=1, Augmix, seed=2021'
# python train.py -m 'se_resnext50_32x4d_004' -c '256, 5fold, tta=1, Augmix, epoch=12, seed=2021'

# python train.py -m 'resnest50_001' -c 'test'
# python train.py -m 'resnest50_002' -c '384, 1fold, tta=1, Augmix, target_encoding_size'

python train.py -m 'resnest50_frelu_001' -c '358 test'

# python ensemble.py -m 'ensemble_001' -c '3models'
# python ensemble.py -m 'ensemble_002' -c '3models, rank_average'
# python ensemble.py -m 'ensemble_003' -c '4models, rank_average'
# python ensemble.py -m 'ensemble_004' -c '5models, rank_average'
# python ensemble.py -m 'ensemble_005' -c '4models, rank_average'
# python ensemble.py -m 'ensemble_006' -c '5models, rank_average'
# python ensemble.py -m 'ensemble_007' -c '6models, rank_average'
# python ensemble.py -m 'ensemble_008' -c '7models, rank_average'
# python ensemble.py -m 'ensemble_009' -c '7models, rank_average'
# python ensemble.py -m 'ensemble_010' -c '2models, rank_average'
# python ensemble.py -m 'ensemble_011' -c '2models, rank_average'
# python ensemble.py -m 'ensemble_012' -c '3models, rank_average'
# python ensemble.py -m 'ensemble_013' -c '4models, rank_average'
# python ensemble.py -m 'ensemble_014' -c '5models, rank_average'
# python ensemble.py -m 'ensemble_015' -c '4models, rank_average'

cd ../
git add -A
git commit -m '...'
git push origin master