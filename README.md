# DA-WSOL_CVPR2022

## Overview
Official implementation of the paper  ``[Weakly Supervised Object Localization as Domain Adaption][paper_url]" (CVPR'22) 

## Assisting WSOL with Domain Adaption

## Network Structure

## Getting Starte

### Prepare the dataset

1. Downloading the train/test split for CUB-200, OpenImages, ILSVRC dataset from [our google drive][meta_url]. 

2. Dowinloading these three datasets from their website: 
     CUB-200 dataset: [CUB-200][cub]
     OpenImages dataset: [OpenImages][open]
     ILSVRC dataset: [ILSVRC][ilsvrc]

3. Putting these three dataset into "$dataroot" with following layout:

<br/>|--CUB <\br>
<br/>|    |--001.Black_footed_Albatross<\br>
<br/>|    |--002.Laysan_Albatross<\br>
<br/>|    |---....<\br>
<br/>|    <\br>
<br/>|--CUBMask<\br>
<br/>|    |--imgs<\br>
<br/>|    |  |--001.Black_footed_Albatross<\br>
<br/>|    |  |--002.Laysan_Albatross<\br>
<br/>|    |  |--....<\br>
<br/>|    |<enter>
<br/>|    |--masks<\br>
<br/>|       |--001.Black_footed_Albatross<\br>
<br/>|       |--002.Laysan_Albatross<\br>
<br/>|       |--....<\br>
<br/>|  <\br>
<br/>|--OpenImages<\br>
<br/>|   |--train<\br>
<br/>|   |   |--01226z<\br>
<br/>|   |   |--018xm<\br>
<br/>|   |   |--....<\br>
<br/>|   |--val<enter>
<br/>|   |   |--01226z<\br>
<br/>|   |   |--018xm<\br>
<br/>|   |   |--....<\br>
<br/>|   |--test<\br>
<br/>|       |--01226z<\br>
<br/>|       |--018xm<\br>
<br/>|       |--....<\br>
<br/>|   <\br>
<br/>|--ILSVRC<\br>
<br/>    |--train<\br>
<br/>    |   |---n01440764<\br>
<br/>    |   |---01443537<\br>
<br/>    |   |---...<\br>
<br/>    |--val
<br/>        |--ILSVRC2012_val_00000001.JPEG<\br>
<br/>        |--ILSVRC2012_val_00000002.JPEG<\br>
<br/>        |--....<\br>

### Training our DA-WSOL

bash run_train.sh

### Testing our DA-WSOL

1. Downloading our checkpoint from [our google drive][checkpoint_url]. 


### Citation


### Acknowledgement
This code and our experiments are conducted based on the release code of [wsolevaluation][EVAL_url] / [transferlearning][tl_url]. Here we thank for their remarkable works.

[EVAL_url]: https://github.com/clovaai/wsolevaluation
[tl_url]: https://github.com/jindongwang/transferlearning


[paper_url]: https://arxiv.org/abs/2203.01714

