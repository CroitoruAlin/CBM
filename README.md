# CBM: Curriculum by Masking (accepted at ECAI 2024)

CBM: Curriculum by Masking implementation repository for reproducing the experiments.

### Data

The data sets (CIFAR-10, CIFAR-100, Tiny ImageNet-200, Food101, Sport Balls, Sea Animals, Architectural Heritage Elements) should be stored in the **data** folder. The CIFAR variants are gathered automatically upon executing the code. The Tiny ImageNet-200 data set can be found at the following address: http://cs231n.stanford.edu/tiny-imagenet-200.zip.

### Other prerequisites

1) The weights for any pre-trained models must be stored in the **pretrained_models** folder. The weights used in our experiments are **CvT-13-224x224-IN-1k.pth** and can be found at: https://onedrive.live.com/?authkey=%21AMXesxbtKwsdryE&id=56B9F9C97F261712%2115004&cid=56B9F9C97F261712.

2) CBM will save the best model in the folder **saved_models**, which has to be created prior to training.

3) Install the Python libraries with the following command:
```sh
    pip install -r requirements.txt
```

### Run

The experiments for our best results can be run with the following command:
```sh
    python main.py --model_name *Model* --dataset *Dataset*
```
1) The notations for the models are: **resnet18**, **wresnet**, **cvt_pretrained**.

2) The notations for the data sets are: **cifar10**, **cifar100**, **tinyimagenet**, **food101**, **balls**, **seaanimals**, **ahe**.
```sh
    python main.py --model_name resnet18 --dataset cifar10
```

### Novelty

The **MaskIn Block** can be found as a layer in all models which is responsible for masking the images based on gradient magnitudes.

The gradient magnitudes are created in the DataLoaders. The implementation is represented by the **get_probability()** function in **data_handlers.py**.

The curricula for our best results are found in the **experiments** folder and the ablation cases in the **ablation** folder.

### References

The ResNet-18 and Wide-ResNet-50 implementation are gathered from the official Curriculum by Smoothing github repository at: https://github.com/pairlab/CBS. 

The CvT-13 implementation is taken from the official github repository at: https://github.com/microsoft/CvT. 
