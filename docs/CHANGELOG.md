## Changelog

### v0.1.0 (22/01/2022)

#### Highlight
* Support various handcrafted mixup methods, AutoMix, and SAMix.
* Support various self- & semi-supervised learning methods.
* Provide image classification benchmarking results of mixup (updating). 

#### Bug Fixes
* Fix bugs of code refactoring.

#### OpenSelfSup (v0.3.0, 14/10/2020) Supported Features

* Mixed Precision Training
* Improvement of GaussianBlur doubles the training speed of MoCo V2, SimCLR, BYOL
* More benchmarking results, including Places, VOC, COCO
* Fix bugs in moco v2 and byol, now the results are reproducible.
* Separate train and test scripts in linear/semi evaluation.
* Support semi-supevised benchmarks: benchmarks/dist_train_semi.sh.
* Move benchmarks related configs into configs/benchmarks/.
* Provide benchmarking results and model download links.
* Support updating network every several interations.
* Support LARS and LAMB optimizer with nesterov (LAMB from MMclassification).
* Support excluding specific parameters from LARS adaptation and weight decay required in SimCLR and BYOL.
