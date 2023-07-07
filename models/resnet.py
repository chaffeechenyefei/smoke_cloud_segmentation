from models.resnet_modules import resnet34, resnet34mlu


def fit_resnet34(pretrained_path=None, class_num = 10):
    model = resnet34(pretrained_path=pretrained_path)
    model.change_last_layer(num_classes=class_num)
    return model

def fit_resnet34mlu(class_num=2):
    model = resnet34mlu(class_num)
    return model


