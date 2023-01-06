# Visual feature extraction

!["our feature extraction frameworks"](https://github.com/Xuplussss/Applying-Segment-Level-Attention-on-Bi-Modal-Transformer-Encoder-for-Audio-Visual-EmotionRecognition/blob/main/visual_feature_extraction/VGGNet.png?raw=true)

## Prepare your data

You need to prepare your file-tuning data as '.h5' file and copy it into 'data/data.h5'. We follow the same data format as 'FER2013'. The file content includes:

```
'Training_pixel' = array([data_number*pixel_number])
'Training_label' = array([data_number])
'PublicText_pixel' 
'PublicText_label' 
'PrivateText_pixel'
'PrivateText_label'
```
for example
```
>>> data['Training_pixel']
<HDF5 dataset "Training_pixel": shape (148493, 2304), type "|u1">
>>> data['Training_pixel'][0] 
array([ 39,  51,  62, ..., 137, 143, 140], dtype=uint8)
>>> data['Training_label'][0] 
0
```
where this data includes 148493 pairs of (images, labels), and the image includes 48*48 pixels. Note that Taining_pixel and Training_label should not be empty.

## Fine tune the model
We provide three methods to obtain visual feature extraction model. It includes 'training from scratch', 'fine-tuning the classifier layer only' and 'fine-tuning the whole model'.


- train from scratch:  
`>>>  python scratchtrain.py --dataset [your data name]`

- fine-tuning the classifier layer only:  
`>>> python finetuneVGG_CL.py --dataset [your data name]`

- fine-tuning the whole model:  
`>>> python finetuneVGG_all_parameter.py --dataset [your data name]`

The fine-tuned models would be saved in '[your data nume]_VGG19/' and the results would be saved in 'result/VGG19/'

## Our fine-tuning losses on BAUM-1 corpus
!["our fine-tuning losses on BAUM-1 corpus"](https://github.com/Xuplussss/Applying-Segment-Level-Attention-on-Bi-Modal-Transformer-Encoder-for-Audio-Visual-EmotionRecognition/blob/main/visual_feature_extraction/TrainingLoss.png?raw=true)