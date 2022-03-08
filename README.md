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

|--CUB <enter>
|    |--001.Black_footed_Albatross<enter>
|    |--002.Laysan_Albatross<enter>
|    |---....<enter>
|    <enter>
|--CUBMask<enter>
|    |--imgs<enter>
|    |  |--001.Black_footed_Albatross<enter>
|    |  |--002.Laysan_Albatross<enter>
|    |  |--....<enter>
|    |<enter>
|    |--masks<enter>
|       |--001.Black_footed_Albatross<enter>
|       |--002.Laysan_Albatross<enter>
|       |--....<enter>
|  <enter>
|--OpenImages<enter>
|   |--train<enter>
|   |   |--01226z<enter>
|   |   |--018xm<enter>
|   |   |--....<enter>
|   |--val<enter>
|   |   |--01226z<enter>
|   |   |--018xm<enter>
|   |   |--....<enter>
|   |--test<enter>
|       |--01226z<enter>
|       |--018xm<enter>
|       |--....<enter>
|   <enter>
|--ILSVRC<enter>
    |--train<enter>
    |   |---n01440764<enter>
    |   |---01443537<enter>
    |   |---...<enter>
    |--val
        |--ILSVRC2012_val_00000001.JPEG<enter>
        |--ILSVRC2012_val_00000002.JPEG<enter>
        |--....<enter>

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

