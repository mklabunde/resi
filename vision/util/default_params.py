from __future__ import annotations

from vision.util import data_structs as ds


def get_default_parameters(architecture_name: str, dataset: ds.Dataset | str) -> ds.Params:
    if isinstance(dataset, str):
        dataset = ds.Dataset(dataset)

    if dataset in [
        ds.Dataset.CIFAR10,
        ds.Dataset.CDOT100,
        ds.Dataset.CDOT50,
        ds.Dataset.CDOT75,
        ds.Dataset.CDOT0,
        ds.Dataset.CDOT25,
        ds.Dataset.GaussMAX,
        ds.Dataset.GaussL,
        ds.Dataset.GaussM,
        ds.Dataset.GaussS,
        ds.Dataset.GaussOff,
        ds.Dataset.RandomLabelC10,
        ds.Dataset.C100CDOT0,
        ds.Dataset.C100CDOT25,
        ds.Dataset.C100CDOT50,
        ds.Dataset.C100CDOT75,
        ds.Dataset.C100CDOT100,
        ds.Dataset.C100GaussMAX,
        ds.Dataset.C100GaussL,
        ds.Dataset.C100GaussM,
        ds.Dataset.C100GaussS,
        ds.Dataset.C100GaussOff,
        ds.Dataset.C100RLABEL100,
        ds.Dataset.C100RLABEL75,
        ds.Dataset.C100RLABEL50,
        ds.Dataset.C100RLABEL25,
        ds.Dataset.CIFAR100,
    ]:
        params = ds.Params(
            architecture_name=architecture_name,
            num_epochs=200,
            save_last_checkpoint=True,
            batch_size=128,
            label_smoothing=False,
            label_smoothing_val=0.1,
            cosine_annealing=True,
            gamma=0.1,
            learning_rate=0.1,
            momentum=0.9,
            nesterov=True,
            weight_decay=5e-4,
            split=0,
            dataset=dataset.value,
        )
    elif dataset == ds.Dataset.DermaMNIST:
        params = ds.Params(
            architecture_name=architecture_name,
            num_epochs=200,  # 250,
            save_last_checkpoint=True,
            batch_size=128,
            label_smoothing=False,
            label_smoothing_val=0.1,
            cosine_annealing=True,
            gamma=0.1,
            learning_rate=0.1,
            momentum=0.9,
            nesterov=True,
            weight_decay=5e-4,
            split=0,
            dataset=dataset.value,
        )
    elif dataset in [
        ds.Dataset.IMAGENET,
        ds.Dataset.IMAGENET100,
        ds.Dataset.TinyIMAGENET,
        ds.Dataset.INCDOT0,
        ds.Dataset.INCDOT25,
        ds.Dataset.INCDOT50,
        ds.Dataset.INCDOT75,
        ds.Dataset.INCDOT100,
        ds.Dataset.INGaussMAX,
        ds.Dataset.INGaussL,
        ds.Dataset.INGaussM,
        ds.Dataset.INGaussS,
        ds.Dataset.INGaussOff,
        ds.Dataset.INRLABEL100,
        ds.Dataset.INRLABEL75,
        ds.Dataset.INRLABEL50,
        ds.Dataset.INRLABEL25,
    ]:
        """Hyperparams taken from
         https://github.com/tensorflow/tpu/tree/master/models/official/resnet
         /resnet_rs/configs
        For VGG16/19/ResNet34/ResNet50/DenseNet121 the resnetrs50 were used.
        For DenseNet161/ResNet101 the ResNetRs101 is used.
        """
        if architecture_name in ["ViT_B16", "ViT_B32", "ViT_L16", "ViT_L32"]:
            params = ds.Params(
                architecture_name=architecture_name,
                num_epochs=300,
                save_last_checkpoint=True,
                batch_size=512,
                label_smoothing=True,
                label_smoothing_val=0.1,
                cosine_annealing=True,
                gamma=0.1,
                learning_rate=3e-3,
                momentum=0.9,
                nesterov=True,
                weight_decay=0.1,
                split=0,
                dataset=dataset.value,
                gradient_clip=1,
                optimizer={
                    "name": "adamw",
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                },
            )
        else:
            params = ds.Params(
                architecture_name=architecture_name,
                num_epochs=200,
                save_last_checkpoint=True,
                batch_size=128,
                label_smoothing=True,
                label_smoothing_val=0.1,
                cosine_annealing=True,
                gamma=0.1,
                learning_rate=0.1,
                momentum=0.9,
                nesterov=True,
                weight_decay=4e-5,
                split=0,
                dataset=dataset.value,
            )

    elif dataset == ds.Dataset.SPLITCIFAR100:
        params = ds.Params(
            architecture_name=architecture_name,
            num_epochs=1000,
            save_last_checkpoint=True,
            batch_size=128,
            label_smoothing=False,
            label_smoothing_val=0.1,
            cosine_annealing=True,
            gamma=0.1,
            learning_rate=0.1,
            momentum=0.9,
            nesterov=True,
            weight_decay=5e-4,
            split=0,
            dataset=dataset.value,
        )
    else:
        raise NotImplementedError(f"Passed Dataset ({dataset}) not implemented.")
    return params


def get_default_arch_params(dataset: ds.Dataset | str, is_vit: bool) -> dict:
    if isinstance(dataset, str):
        dataset = ds.Dataset(dataset)
    if dataset in [
        ds.Dataset.CIFAR10,
        ds.Dataset.TEST,
        ds.Dataset.CIFAR10,
        ds.Dataset.CDOT100,
        ds.Dataset.CDOT50,
        ds.Dataset.CDOT75,
        ds.Dataset.CDOT0,
        ds.Dataset.CDOT25,
        ds.Dataset.GaussMAX,
        ds.Dataset.GaussL,
        ds.Dataset.GaussM,
        ds.Dataset.GaussS,
        ds.Dataset.GaussOff,
        ds.Dataset.RandomLabelC10,
    ]:
        output_classes = 10
        in_ch = 3
        input_resolution = (32, 32)
        if is_vit:
            input_resolution = (224, 224)
        early_downsampling = False
        global_average_pooling = 4
    elif dataset in [
        ds.Dataset.IMAGENET,
        ds.Dataset.IMAGENET100,
        ds.Dataset.INCDOT0,
        ds.Dataset.INCDOT25,
        ds.Dataset.INCDOT50,
        ds.Dataset.INCDOT75,
        ds.Dataset.INCDOT100,
        ds.Dataset.INGaussMAX,
        ds.Dataset.INGaussL,
        ds.Dataset.INGaussM,
        ds.Dataset.INGaussS,
        ds.Dataset.INGaussOff,
        ds.Dataset.INRLABEL100,
        ds.Dataset.INRLABEL75,
        ds.Dataset.INRLABEL50,
        ds.Dataset.INRLABEL25,
    ]:
        output_classes = 1000 if dataset == ds.Dataset.IMAGENET else 100
        in_ch = 3
        input_resolution = (224, 224)
        early_downsampling = True
        global_average_pooling = 5
    elif dataset in [
        ds.Dataset.CIFAR100,
        ds.Dataset.C100CDOT0,
        ds.Dataset.C100CDOT25,
        ds.Dataset.C100CDOT50,
        ds.Dataset.C100CDOT75,
        ds.Dataset.C100CDOT100,
        ds.Dataset.C100GaussMAX,
        ds.Dataset.C100GaussL,
        ds.Dataset.C100GaussM,
        ds.Dataset.C100GaussS,
        ds.Dataset.C100GaussOff,
        ds.Dataset.C100RLABEL100,
        ds.Dataset.C100RLABEL75,
        ds.Dataset.C100RLABEL50,
        ds.Dataset.C100RLABEL25,
    ]:
        output_classes = 100
        in_ch = 3
        input_resolution = (32, 32)
        if is_vit:
            input_resolution = (224, 224)
        early_downsampling = False
        global_average_pooling = 4
    elif dataset == ds.Dataset.SPLITCIFAR100:
        output_classes = 5  # 20 Splits a 5 Classes
        in_ch = 3
        input_resolution = (32, 32)
        if is_vit:
            input_resolution = (224, 224)
        early_downsampling = False
        global_average_pooling = 4
    elif dataset == ds.Dataset.DermaMNIST:
        output_classes = 7  # 20 Splits a 5 Classes
        in_ch = 3
        input_resolution = (28, 28)
        early_downsampling = False
        global_average_pooling = 4
    elif dataset == ds.Dataset.TinyIMAGENET:
        output_classes = 200  # 20 Splits a 5 Classes
        in_ch = 3
        input_resolution = (64, 64)
        early_downsampling = False
        global_average_pooling = 5
    else:
        raise NotImplementedError(f"Unexpected dataset! Got {dataset}!")

    return dict(
        n_cls=output_classes,
        in_ch=in_ch,
        input_resolution=input_resolution,
        early_downsampling=early_downsampling,
        global_average_pooling=global_average_pooling,
    )
