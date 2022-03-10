# DA-WSOL_CVPR2022

## Overview
Official implementation of the paper  ``[Weakly Supervised Object Localization as Domain Adaption][paper_url]" (CVPR'22) 

## Assisting WSOL with Domain Adaption

## Network Structure

## Getting Starte

### Prepare the dataset

1. Downloading the train/test split and bounding-boxes annotation for CUB-200, OpenImages, ILSVRC dataset from [our google drive][meta_url]. 

2. Dowinloading source images and pixel-level localiztion masks of these three datasets from: 
     <br/>CUB-200 dataset: [source images][cub_image_url],  [segmentation mask][cub_mask_url]
     <br/>OpenImages dataset: [source images][open_image_url], [segmentation mask][open_mask_url]
     <br/>ILSVRC dataset: [source_images][ilsvrc_url]

3. Putting these three dataset into "$dataroot" with following layout:
 
     <br/>|--CUB 
     <br/>|    |--001.Black_footed_Albatross
     <br/>|    |--002.Laysan_Albatross
     <br/>|    |--....

     <br/>|--CUBMask
     <br/>|    |--imgs
     <br/>|    |  |--001.Black_footed_Albatross
     <br/>|    |  |--002.Laysan_Albatross
     <br/>|    |  |--....
     <br/>|    |
     <br/>|    |--masks
     <br/>|    |   |--001.Black_footed_Albatross
     <br/>|    |   |--002.Laysan_Albatross
     <br/>|    |   |--....

     <br/>|--OpenImages
     <br/>|   |--train
     <br/>|   |   |--01226z
     <br/>|   |   |--018xm
     <br/>|   |   |--....
     <br/>|   |--val
     <br/>|   |   |--01226z
     <br/>|   |   |--018xm
     <br/>|   |   |--....
     <br/>|   |--test
     <br/>|   |    |--01226z
     <br/>|   |    |--018xm
     <br/>|   |    |--....

     <br/>|--ILSVRC
     <br/>    |--train
     <br/>    |   |---n01440764
     <br/>    |   |---01443537
     <br/>    |   |---...
     <br/>    |--val
     <br/>    |    |--ILSVRC2012_val_00000001.JPEG
     <br/>    |    |--ILSVRC2012_val_00000002.JPEG
     <br/>    |    |--....

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
[checkpoint_url]: https://drive.google.com/drive/folders/1NLrTq8kllz46ESfBSWJFZ638PKPDXLQ1?usp=sharing
[meta_url]: https://drive.google.com/drive/folders/1xQAjoLyD96vRd6OSF72TAGDdGOLVJ0yE?usp=sharing
[cub_image_url]: https://drive.google.com/file/d/1U6cwKHS65wayT9FFvoLIA8cn1k0Ot2M1/view?usp=drive_open
[open_mask_url]: https://drive.google.com/file/d/1KZQLpwkuF0HgmJ04P9N9lmYvvGU9-ACP/view?usp=sharing
[open_image_url]: https://drive.google.com/file/d/1oOb4WQ-lb8SYppHEg3lWnpHk1X8WVO4e/view
[open_mask_url]: https://drive.google.com/file/d/1eu1YvcZlsEalhXTS_5Ni5tkImCliIPie/view
[ilsvrc_url]: https://image-net.org

