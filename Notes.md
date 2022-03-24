# Notes

## Image encoder:
- models: resnet 18, resnet 34, resnet 50
- norms: batch norm, instance norm, layer norm, (later: group norm with different group #'s)
- pretrained weights
- projection?
- learning rate scheduler
- dropout

## Text encoder:
- num transformer layers:
- num attention heads:
- pre vs post LN
- positional encoding
- lr scheduler 
- warm up stage: T_warmup, lr_max (5e-4 or 1e-3 (Adam), 5e-3 or 1e-3 (SGD))
- dropout
- TRAIN WITH DT-Fixup?

## Optimizer:
- Betas (0.9, 0.98)

## EBL:
- RMSE

## SF
- accuracy, misclassification

## Save models

- different learning rates for resnet and transformer

- data augmentation?