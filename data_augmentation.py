import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
import os
import re
from skimage import io


def random_rotation(image_array: ndarray, r):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    degree = r
    return sk.transform.rotate(image_array, degree)

def horizontal_flip(image_array: ndarray, r):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

def affine_transform(image_array: ndarray, r):
    tform = sk.transform.AffineTransform(scale=(1.2, 1.2), translation=(0, -100))
    img2 = sk.transform.warp(image_array, tform)
    return img2

def apply_in_folders(root):
    # dictionary of the transformations we defined earlier
    available_transformations = {
        'rotate': random_rotation,
        'horizontal_flip': horizontal_flip,
        'Affine_transformation': affine_transform
    }

    dir_list = next(os.walk(root))[1]
    for first_index in dir_list:
        next_dir = next(os.walk(root+"/"+first_index))[1]
        #print("The directory: "+first_index+" contains "+str(len(next_dir))+" folders")
       
        for folder in next_dir:
            random_degree = random.uniform(-25, 25)
            num_transformations_to_apply = random.randint(1, len(available_transformations))
            key = random.choice(list(available_transformations))
            final_files_folder = os.listdir(root+"/"+first_index+"/"+folder)
            #print("The directory: "+first_index+" has a folder "+str(folder)+" which contains no of files: "+str(len(final_files_folder)))
            transformed_image = None
            if str(folder) != "Videos":
                new_dir_augmented = str(root)+"/"+str(first_index)+"/"+str(folder)+"_augmented"
                os.makedirs(new_dir_augmented)
            for image_file in final_files_folder:
                if image_file.endswith("jpg"):
                    image_to_transform = sk.io.imread(str(root)+"/"+str(first_index)+"/"+str(folder)+"/"+image_file)
                    transformed_image = available_transformations[key](image_to_transform, random_degree)
                    #print(image_file)
                    
                    new_file_path = '%s/augmented_image_%s.jpg' % (new_dir_augmented, str(re.sub('\.jpg$', '', image_file)))
                    #print(new_file_path)
                    #write image to the disk
                    io.imsave(new_file_path, transformed_image)
    print("Data has been augmented")
                    
                 