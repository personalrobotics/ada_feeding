# Ada Feeding Perception
This code performs image segmentation using the [Segment Anything](https://github.com/facebookresearch/segment-anything). It takes an input image and a point as input and generates masks for the regions of interest in the image.

## Installation
```
chmod +x install.sh
./install.sh
```

## Test
To run the test code with an example image and input points, use the following command:
```
python test/image_segmentation.py --input_image <example_image.jpg> --input_point [x,y]
```
Here are some examples:
```
python test/test_sam.py --input_image test/food_img/sete_00100.jpg --input_point [800,500]
python test/test_sam.py --input_image test/food_img/sete_00100.jpg --input_point [1050,400]
python test/test_sam.py --input_image test/food_img/sete_00100.jpg --input_point [1200,400]

python test/test_sam.py --input_image test/food_img/setf_00113.jpg --input_point [1050,400]
python test/test_sam.py --input_image test/food_img/setf_00113.jpg --input_point [1300,500]
python test/test_sam.py --input_image test/food_img/setf_00113.jpg --input_point [1300,800]
```