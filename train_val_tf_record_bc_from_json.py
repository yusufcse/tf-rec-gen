#===========================================================================================
#
#title           :TF record generator for bangla charecter detection
#description     :This script read json file and images from a directory to generate TF record. 
#				  It is only applicable for a single bounding box in an image. This script is appropiate for labelme software.
#author		 	 :Abu Yusuf
#date            :20191228
#version         :1.1    
#usage		     :json file parse and generate TF record.
#notes           :Install minimum python 3.6, Python Image Library.
#
#============================================================================================

import sys
import os
import json
import io

import tensorflow as tf
from object_detection.utils import dataset_util


from PIL import Image  # Python image library

os.chdir(os.getcwd())

labelFolder="bc_labels"  # name of the label folder that contains json files
imageFolder="bc_images" # name of the image folder that contains image files

recordFolder="tfRecord/bc_train.record" # path to the train record folder
valRecordFolder="tfRecord/bc_val.record" # path to the validation folder

# Define flags 
flags = tf.app.flags
flags.DEFINE_string('output_path', recordFolder, 'Path to output TFRecord')
flags.DEFINE_string('output_valpath', valRecordFolder, 'Path to output TFRecord')
FLAGS = flags.FLAGS

def create_tf_record(data):

    # TODO(user): Populate the following variables from your data.
    height = data["height"] # Image height
    width = data["width"] # Image width
    filename = data["filename"].encode('utf8') # Filename of the image. Empty if image is not from file
    
           
    img_path = imageFolder+os.sep+data["filename"]
    
    
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_image_data = fid.read()
        
    image_format =  'png' # b'jpeg' or b'png'
    
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    
    
    x_y_coordinates = data["bbox"]
    xmins.append(float(x_y_coordinates[0][0]) / width) # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs.append(float(x_y_coordinates[1][0]) / width) # List of normalized right x coordinates in bounding box(1 per box)
    ymins.append(float(x_y_coordinates[0][1]) / height) # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs.append(float(x_y_coordinates[1][1]) / height) # List of normalized bottom y coordinates in bounding box (1 per box)
        
    #classes_text.append(x_y_coordinates["identity"].encode('utf8')) # List of string class name of bounding box (1 per box)
        
    #if data["identity"] == "soreo":
    classes.append(1)
    classes_text.append("soreo".encode('utf8'))
        
        
      
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
	
    return tf_example


  
def main(_):
    
    
    writer = tf.io.TFRecordWriter(FLAGS.output_path)    
       
    # TODO(user): Write code to read in your dataset to examples variable
    
    jsonFiles=os.listdir(labelFolder); # reading directory
    
    # Read directory contains json files one by one
    for file in jsonFiles:
        
        # Read string of a json file
        with open(labelFolder+os.sep+file) as jsonString: 
            
            json_data = json.load(jsonString)
            #print(json_data['imagename'])
            
            imInfo = Image.open(imageFolder+os.sep+json_data["imagePath"])
            w, h = imInfo.size
           
            # print('width: ', w)
            # print('height:', h)
            # print(json_data["children"][0]["mincol"])
            # print(json_data["children"][0]["minrow"])
            # print(json_data["children"][0]["maxcol"])
            # print(json_data["children"][0]["maxrow"])
                        
            dataset = {
                "filename": json_data['imagePath'],
                "width":w,
                "height":h,
                "bbox":json_data['shapes'][0]["points"],
				"identity":json_data['shapes'][0]["label"],
            }
            
           
            #print(dataset["bbox"])
                                    
            #for example in examples:
            tf_record = create_tf_record(dataset)
            writer.write(tf_record.SerializeToString())
            
    
    writer.close()
    
    
if __name__ == '__main__':
    tf.app.run()