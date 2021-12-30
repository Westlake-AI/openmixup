# reference: https://github.com/jacobgil/pytorch-grad-cam
import os
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models


class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)
        
        return target_activations, x


def preprocess_image(img):
    # using ImageNet mean, std
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(img, mask, save_name, save_path):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("{}/{}_cam.jpg".format(save_path, save_name), np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)
        
        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)
        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply
        
        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None, target=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        cls_res = {}  # classification False or True
        if index == None:
            index = np.argmax(output.cpu().data.numpy())
            if target != None:
                assert isinstance(target, int) or isinstance(target, np.int)
                cls_res["bool"] = index == target
                print("cls={}, index={}, target={}".format(cls_res["bool"], index, target))
                cls_res["pred"] = index

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output, cls_res


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)


def fit_model_state_dict(model, ckpt_path):
    ckpt = torch.load(ckpt_path)
    ckpt_state_dict = ckpt["state_dict"]
    model_state_dict = model.state_dict()
    print( len(ckpt_state_dict.keys()), len(model_state_dict.keys()) )

    pretrained = {}
    for k, v in ckpt_state_dict.items():
        if k.find("backbone") != -1:
            new_key = k.split("backbone.")[-1]
        elif k.find("head.") != -1:  # head.fc_cls.weight, head.fc_cls.bias -> 'fc.weight', 'fc.bias'
            new_key = "fc." + k.split(".")[-1]
        else:
            new_key = k
        if new_key in model_state_dict.keys():
            pretrained[new_key] = v
        else:
            print(k)
    print("loaded model keys={}".format(len(pretrained.keys())))

    model_state_dict.update(pretrained)
    model.load_state_dict(model_state_dict)


def get_sample_cub200_dict():
    return {
        "001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg": 0,
        "001.Black_footed_Albatross/Black_Footed_Albatross_0002_55.jpg": 0,
        "002.Laysan_Albatross/Laysan_Albatross_0002_1027.jpg": 1,
        "002.Laysan_Albatross/Laysan_Albatross_0082_524.jpg": 1,
        "003.Sooty_Albatross/Sooty_Albatross_0003_1078.jpg": 2,
        "003.Sooty_Albatross/Sooty_Albatross_0044_1105.jpg": 2,
        "004.Groove_billed_Ani/Groove_Billed_Ani_0068_1538.jpg": 3,
        "004.Groove_billed_Ani/Groove_Billed_Ani_0094_1540.jpg": 3,
        "005.Crested_Auklet/Crested_Auklet_0006_1813.jpg": 4,
        "005.Crested_Auklet/Crested_Auklet_0042_794902.jpg": 4,
        "006.Least_Auklet/Least_Auklet_0014_1901.jpg": 5,
        "006.Least_Auklet/Least_Auklet_0032_795068.jpg": 5,
        "007.Parakeet_Auklet/Parakeet_Auklet_0017_795924.jpg": 6,
        "007.Parakeet_Auklet/Parakeet_Auklet_0072_795929.jpg": 6,
        "008.Rhinoceros_Auklet/Rhinoceros_Auklet_0042_2101.jpg": 7,
        "008.Rhinoceros_Auklet/Rhinoceros_Auklet_0027_797496.jpg": 7,
        "009.Brewer_Blackbird/Brewer_Blackbird_0099_2560.jpg": 8,
        "009.Brewer_Blackbird/Brewer_Blackbird_0106_2608.jpg": 8,
        "010.Red_winged_Blackbird/Red_Winged_Blackbird_0022_4483.jpg": 9,
        "010.Red_winged_Blackbird/Red_Winged_Blackbird_0017_4116.jpg": 9,
    }


def get_sample_pets37_dict():
    return {
        "Abyssinian_201.jpg": 0,
        "Abyssinian_202.jpg": 0,
        "american_bulldog_203.jpg": 1,
        "american_bulldog_205.jpg": 1,
        "american_pit_bull_terrier_191.jpg": 2,
        "american_pit_bull_terrier_192.jpg": 2,
        "basset_hound_191.jpg": 3,
        "basset_hound_192.jpg": 3,
        "beagle_195.jpg": 4,
        "beagle_196.jpg": 4,
        "Bengal_192.jpg": 5,
        "Bengal_193.jpg": 5,
        "Birman_191.jpg": 6,
        "Birman_192.jpg": 6,
        "Bombay_200.jpg": 7,
        "Bombay_201.jpg": 7,
        "boxer_191.jpg": 8,
        "boxer_192.jpg": 8,
        "British_Shorthair_212.jpg": 9,
        "British_Shorthair_213.jpg": 9,
    }


def get_sample_dogs120_dict():
    return {
        "n02085620-Chihuahua/n02085620_2650.jpg": 0,
        "n02085620-Chihuahua/n02085620_4919.jpg": 0,
        "n02085782-Japanese_spaniel/n02085782_191.jpg": 1,
        "n02085782-Japanese_spaniel/n02085782_1929.jpg": 1,
        "n02085936-Maltese_dog/n02085936_2760.jpg": 2,
        "n02085936-Maltese_dog/n02085936_8350.jpg": 2,
        "n02086079-Pekinese/n02086079_14999.jpg": 3,
        "n02086079-Pekinese/n02086079_6333.jpg": 3,
        "n02086240-Shih-Tzu/n02086240_5096.jpg": 4,
        "n02086240-Shih-Tzu/n02086240_1690.jpg": 4,
        "n02086646-Blenheim_spaniel/n02086646_23.jpg": 5,
        "n02086646-Blenheim_spaniel/n02086646_2834.jpg": 5,
        "n02086910-papillon/n02086910_2670.jpg": 6,
        "n02086910-papillon/n02086910_7766.jpg": 6,
        "n02087046-toy_terrier/n02087046_430.jpg": 7,
        "n02087046-toy_terrier/n02087046_5843.jpg": 7,
        "n02087394-Rhodesian_ridgeback/n02087394_1963.jpg": 8,
        "n02087394-Rhodesian_ridgeback/n02087394_3619.jpg": 8,
        "n02088094-Afghan_hound/n02088094_7106.jpg": 9,
        "n02088094-Afghan_hound/n02088094_12664.jpg": 9,
    }



if __name__ == '__main__':
    """ Usage of grad-CAM visualization
        version 01.05

    Settings:
        class_num: set the class num of your dataset
        data_name: choose a dataset
        model_arch: CNN arth such as "resnet50" in torchvision
        model_name: name to save
        model_path: path of linear_classification checkpoint
        img_path: a dict of img_name (string) and img_class (int)
        img_root: dirs of image dataset
    
    Running:
        1. set a config and data_name.
        2. python tools/gradcam.py
    """

    args = {
        # ----- dataset 1: cub200 ------
        "class_num": 200,
        "data_name": "cub200",
        # ------ dataset 2: pets -------
        # "class_num": 37,
        # "data_name": "pets37",
        # ----- dataset 3: dogs120 -----
        # "class_num": 120,
        # "data_name": "dogs120",
        # -------- cnn arch ----------
        "model_arch": "resnet50",
        # ------------------------- Example: SL imagenet path -----------------------------
        "model_name": "imagenet_r50_pytorch",  # model name 1
        "model_path": "./work_dirs/benchmarks/linear_classification/cub200/r50_last_2gpu_cub200/imagenet_r50_pytorch.pth/epoch_100.pth",  # cub200 baseline
        # "model_path": "./work_dirs/benchmarks/linear_classification/pets/r50_last_2gpu_pets/imagenet_r50_pytorch.pth/epoch_100.pth",  # Pets baseline
        # "model_path": "./work_dirs/benchmarks/linear_classification/dogs120/r50_last_2gpu_dogs120/imagenet_r50_pytorch.pth/epoch_100.pth",  # Dogs baseline
        # "model_name": "imagenet_r50_pytorch_bbox",  # model name 2
        # "model_path": "./work_dirs/benchmarks/linear_classification/cub200/r50_last_2gpu_cub200_ablation/imagenet_r50_pytorch.pth/epoch_100.pth",  # cub200, no back: bbox
        # "model_name": "imagenet_r50_pytorch_mask",  # model name 3
        # "model_path": "./work_dirs/benchmarks/linear_classification/pets/r50_last_2gpu_pets_ablation/imagenet_r50_pytorch.pth/epoch_100.pth",  # pets, no back: mask
        # -------- image dict ----------
        "img_path": dict(),
        # ------- dataset root ---------
        # "img_root": "/usr/commondata/public/CUB200/CUB_200/images/",  # CUB-200, ori img
        # "img_root": "/usr/commondata/public/CUB200/CUB_200/images_bbox/",  # CUB-200, no back: bbox
        # "img_root": "/usr/commondata/public/Pets37/images/",  # Pets-37, ori img
        "img_root": "/usr/commondata/public/Pets37/images_segmented/",  # Pets-37, no back: mask
        # "img_root": "/usr/commondata/public/Dogs120/Images/",  # Dogs-120, ori img
        "use_cuda": torch.cuda.is_available()
    }
    assert args["data_name"] in ["cub200", "pets37", "dogs120"]

    # get sample dict
    try:
        args["img_path"] = eval("get_sample_{}_dict()".format(args["data_name"]))
    except:
        print("Please add 'get_sample_{}_dict()' to load img list!".format(args["data_name"]))
        exit()
    
    i = 0
    for k,v in args["img_path"].items():
        
        cnn_arch = models.__dict__[args["model_arch"]]
        model = cnn_arch(num_classes=args["class_num"])
        fit_model_state_dict(model, args["model_path"])
        grad_cam = GradCam(
            model=model, feature_module=model.layer4, target_layer_names=["2"], use_cuda=args["use_cuda"])
        print(i)

        if len(args["img_root"]) > 1:
            img_path = args["img_root"] + k
        else:
            img_path = k
        
        img = cv2.imread(img_path, 1)
        img = np.float32(cv2.resize(img, (224, 224))) / 255
        input = preprocess_image(img)

        # save to plot dir
        save_path = "work_dirs/plot_cam/{}/{}".format(args["data_name"], args["model_name"])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # choose a target label, set None as default
        target_index = None
        mask = grad_cam(input, target_index)

        gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args["use_cuda"])

        gb, cls_res = gb_model(input, index=target_index, target=v)
        gb = gb.transpose((1, 2, 0))
        cam_mask = cv2.merge([mask, mask, mask])
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)

        cls_res_str = str(cls_res["bool"]) if cls_res["bool"] else "{}_pred{}".format(str(cls_res["bool"]), str(cls_res["pred"]))
        show_cam_on_image(img, mask, "{}_C{}_id{}_{}".format(args["model_name"], str(v), str(i), cls_res_str), save_path)
        # cv2.imwrite('{}/{}_id{}_gb.jpg'.format(save_path, args["model_name"], str(i)), gb)
        cv2.imwrite('{}/{}_C{}_id{}_{}_cam_gb.jpg'.format(save_path, args["model_name"], str(v), str(i), cls_res_str), cam_gb)
        i += 1