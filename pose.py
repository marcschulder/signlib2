import os, sys
import subprocess

def run_openpose(openpose_dir, root):
    #root = "."+root
    first_dir= os.listdir(root) # get all files' and folders' names in the current directory
    
    for direc in first_dir:
        second_dir = root+"/"+direc
        second_dir_dir = os.listdir(second_dir)
        for final_dir in second_dir_dir:
            if (final_dir != "Videos"):
                directory_to_save = second_dir+"/"+final_dir
                final_directory = str(directory_to_save)
                print(final_directory)
                filename = "%s --image_dir %s --write_images %s --display 0 --model_pose COCO --write_json %s" %(openpose_dir, final_directory, final_directory, final_directory)
                
                #subprocess.call(["start",filename], shell=True)
                
                os.system("start /wait cmd /k "+filename)


def run_test(openpose_dir, root):
    
    filename = "%s --image_dir %s --write_images %s --display 0 --model_pose COCO --write_json %s" %(openpose_dir, root, root, root)
                
            
    os.system("start /wait cmd /k "+filename)
    
#RUN THE ABOVE IN JUPYTER NOTEBOOK IN CASE ABOVE DOESN'T WORK!   
    
# import os
# root = "./Augmented_data/"
# first_dir= os.listdir(root) # get all files' and folders' names in the current directory

# for direc in first_dir:
#     second_dir = root+"/"+direc
#     second_dir_dir = os.listdir(second_dir)
#     for final_dir in second_dir_dir:
#         if (final_dir != "Videos"):
#             directory_to_save = second_dir+"/"+final_dir
#             final_directory = str(directory_to_save)
#             print(final_directory)
#             !E:\Testing_signlib\openpose\bin\OpenPoseDemo.exe --image_dir $final_directory --write_images $final_directory --display 0 --model_pose COCO --write_json $final_directory 
