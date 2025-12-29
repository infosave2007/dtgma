# DTG-MA CNN Experiment Results (RESNET backbone)

**Device**: mps
**Backbone**: RESNET
**Epochs**: 10
**Date**: 2025-12-29 20:01


## Split MNIST

| Method | Accuracy | Forgetting | Time | Params |
|--------|----------|------------|------|--------|
| DTG-MA+RESNET | 77.5% | 0.0% | 347.8s | 19,190,986 |
| EWC+RESNET | 65.9% | 42.1% | 1118.4s | 11,168,706 |
| DER+++RESNET | 62.4% | 46.4% | 1490.6s | 11,168,706 |
| HAT+RESNET | 60.3% | 45.3% | 1161.6s | 11,433,922 |
| PackNet+RESNET | 59.1% | 50.5% | 1013.4s | 11,431,362 |
| Fine-tune+RESNET | 58.6% | 51.1% | 1009.5s | 11,168,706 |
