from experiments import resnet_experiments, wresnet_experiments, cvt_pretrained_experiments

RUNS = {
    'resnet18': {
        'cifar10': resnet_experiments.train_cifar10,
        'cifar100': resnet_experiments.train_cifar100,
        'tinyimagenet': resnet_experiments.train_tinyimagenet,
	'food101' : resnet_experiments.train_food101,
    },
    'wresnet': {
        'cifar10': wresnet_experiments.train_cifar10,
        'cifar100': wresnet_experiments.train_cifar100,
        'tinyimagenet': wresnet_experiments.train_tinyimagenet,
	'food101' : wresnet_experiments.train_food101,    
    },
    'cvt_pretrained': {
        'cifar10': cvt_pretrained_experiments.train_cifar10,
        'cifar100': cvt_pretrained_experiments.train_cifar100,
        'tinyimagenet': cvt_pretrained_experiments.train_tinyimagenet,
	'food101' : cvt_pretrained_experiments.train_food101, 
    'balls' :    cvt_pretrained_experiments.train_balls,
    'seaanimals' : cvt_pretrained_experiments.train_seaanimals,
    'ahe' : cvt_pretrained_experiments.ahe_train_1,
    },
}
