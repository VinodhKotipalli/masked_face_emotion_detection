#!/bin/sh

# change this to reflect absolute path to data sets
DATA_DIR=/CS541-class-project/archive

# mask data producing two sets, one with original and masked in same directory
# another with masked data in own directory sets
for i in $(find $DATA_DIR -maxdepth 2 -type d)
do
   echo $i
   python mask_the_face.py --path $i --mask_type surgical --verbose --write_original_image
done

        
