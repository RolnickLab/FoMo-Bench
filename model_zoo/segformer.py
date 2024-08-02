from transformers import SegformerForSemanticSegmentation
import torch
import torch.nn as nn

backbones = {"mit-b0-ade":"nvidia/mit-b0","mit-b1-ade":"nvidia/segformer-b1-finetuned-ade-512-512","mit-b2-ade":"nvidia/segformer-b2-finetuned-ade-512-512","mit-b3-ade":"nvidia/segformer-b3-finetuned-ade-512-512","mit-b4-ade":"nvidia/segformer-b4-finetuned-ade-512-512","mit-b5-ade":"nvidia/segformer-b5-finetuned-ade-640-640"}

class Segformer(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        if config['backbone'] not in backbones:
            print('Backbone: ',config['backbone'],' Not supported!')
            exit(2)
        self.model = SegformerForSemanticSegmentation.from_pretrained(backbones[config['backbone']],num_labels=config['num_classes'],ignore_mismatched_sizes=True)
        
        if config['in_channels']!=3:
            self.model.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(config['in_channels'],32,kernel_size=(7,7),stride=(4,4),padding=(3,3))

    def forward(self,x):
        logits = self.model(x)['logits']
        upsampled_logits = nn.functional.interpolate(
                                logits,
                                size=(x.shape[2],x.shape[3]), # (height, width)
                                mode='bilinear',
                                align_corners=False)
        
        return upsampled_logits

'''
#Example usage:
#================
config= {'in_channels':5,'num_classes':2,'backbone':'mit-b0-ade'}
k = torch.randn((4,config['in_channels'],224,224))
model = Segformer(config)
print(model)
logits = model(k)
print(logits.shape)'''