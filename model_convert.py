import torch
import model

model_convert_dict={
    'features.0.weight':'basevgg.conv1_1.weight',
    'features.2.weight':'basevgg.conv1_2.weight',
    'features.5.weight':'basevgg.conv2_1.weight',
    'features.7.weight':'basevgg.conv2_2.weight',
    'features.10.weight':'basevgg.conv3_1.weight',
    'features.12.weight':'basevgg.conv3_2.weight',
    'features.14.weight':'basevgg.conv3_3.weight',
    'features.17.weight':'basevgg.conv4_1.weight',
    'features.19.weight':'basevgg.conv4_2.weight',
    'features.21.weight':'basevgg.conv4_3.weight',
    'features.24.weight':'basevgg.conv5_1.weight',
    'features.26.weight':'basevgg.conv5_2.weight',
    'features.28.weight':'basevgg.conv5_3.weight',
    'features.0.bias': 'basevgg.conv1_1.bias',
    'features.2.bias': 'basevgg.conv1_2.bias',
    'features.5.bias': 'basevgg.conv2_1.bias',
    'features.7.bias': 'basevgg.conv2_2.bias',
    'features.10.bias': 'basevgg.conv3_1.bias',
    'features.12.bias': 'basevgg.conv3_2.bias',
    'features.14.bias': 'basevgg.conv3_3.bias',
    'features.17.bias': 'basevgg.conv4_1.bias',
    'features.19.bias': 'basevgg.conv4_2.bias',
    'features.21.bias': 'basevgg.conv4_3.bias',
    'features.24.bias': 'basevgg.conv5_1.bias',
    'features.26.bias': 'basevgg.conv5_2.bias',
    'features.28.bias': 'basevgg.conv5_3.bias',
}

HED = model.HED()

model_dict={}
state_dict=torch.load('vgg16-00b39a1b.pth')
for k,v in state_dict.items():
    if k in model_convert_dict.keys():
        model_dict[model_convert_dict[k]]=state_dict[k]

HED_state_dict=HED.state_dict()
HED_state_dict.update(model_dict)
name='vgg16_convert.pth'
torch.save(HED_state_dict,name)