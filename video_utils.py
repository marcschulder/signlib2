import cv2
import time
import os
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import pympi
import re
from fnmatch import fnmatch
import argparse


def video_to_frames(input_loc, output_loc):
    """
    Function to extract frames from a video.
    Specify: path to video and output path
    """
    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) -1
    print(video_length)
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        # Write the results back to output location.
        resized_frame = cv2.resize(frame, (720, 480), interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), resized_frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds for conversion." % (time_end-time_start) + str("\n\n"))
            break
            
    return(video_length)

def frames_to_video(dir_path, ext, output):
    """
    Function to compile frames into a video.
    Specify directory of images, extention of images ex. png, output name followed by codec ex.mp4
    """
#     dir_path = './Outputs/final/'
#     ext = 'png'
#     output = 'output_video.mp4'

    images = []
    for f in os.listdir(dir_path):
        if f.endswith(ext):
            images.append(f)
    print(len(images))
    # Determine the width and height from the first image
    image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(image_path)
    cv2.imshow('video',frame)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

    for image in images:

        image_path = os.path.join(dir_path, image)
        frame = cv2.imread(image_path)

        out.write(frame) # Write out frame to video

        cv2.imshow('video',frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
            break

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()

    print("The output video is {}".format(output))
    
def extract_videos_from_annotations(root, gloss_list):
    """
    Function to extract videos from eaf annotations.
    Additionally it creates folders with the extracted frames for each video.
    Specify the path of the eaf file and the gloss list.
    ex. ("./Raw_videos/original_video",["NS", "1H", "2H"])
    Make sure the eaf file has the same name as the video
    """
    def check_folders(gl_name):
        directory1 = "./Data/"+gl_name+"/Videos/"
        if not os.path.exists(directory1):
            os.makedirs(directory1)

    # Find the eaf file
    cwd = os.getcwd()
    root = str(cwd)
    pattern = "*.eaf"


    for root, dirs, files in os.walk(root, topdown=False):
        for name in files:
            if fnmatch(name, pattern):
                video_name = re.sub('\.eaf$', '', name)+".mp4"
                print(video_name)
                video_name = root+"/"+video_name
                file = pympi.Eaf(file_path=root+"/"+name)
                tier_names = file.get_tier_names()


                for tier_name in tier_names:           
                    annotations = file.get_annotation_data_for_tier(tier_name)
                    count = 0
                    for annotation in annotations:
                        for gloss in gloss_list:
                            if annotation[2] == gloss:
                                start = annotation[0]
                                end = annotation[1]
                                print(start/1000,end/1000)
                                check_folders(gloss)
                                ffmpeg_extract_subclip(video_name, start/1000, end/1000, targetname="Data/"+str(gloss)+"/Videos/"+"%#05d.mp4" % (count+1))
                                # Comment next line if you don't want to extract the frames for each video
                                video_to_frames("Data/"+str(gloss)+"/Videos/"+"%#05d.mp4" % (count+1), "Data/"+str(gloss)+"/"+"%#05d" % (count+1) )
                                count = count+1
                    if count == 0:
                        print("No annotation found with this name")

def extract_videos_from_annotations_colab(video_name, eaf_file_name, gloss_list):
    """
    Function to extract videos from eaf annotations.
    Additionally it creates folders with the extracted frames for each video.
    """
    def check_folders(gl_name):
        directory1 = "openpose/"+gl_name+"/"
        if not os.path.exists(directory1):
            os.makedirs(directory1)


    file = pympi.Eaf(file_path=eaf_file_name)
    tier_names = file.get_tier_names()


    for tier_name in tier_names:           
        annotations = file.get_annotation_data_for_tier(tier_name)
        count = 0
        for annotation in annotations:
            for gloss in gloss_list:
                if annotation[2] == gloss:
                    start = annotation[0]
                    end = annotation[1]
                    print(start/1000,end/1000)
                    check_folders(gloss)
                    ffmpeg_extract_subclip(video_name, start/1000, end/1000, targetname="openpose/"+str(gloss)+"/"+"%#05d.mp4" % (count+1))
                    # Comment next line if you don't want to extract the frames for each video
                    # video_to_frames("Data/"+str(gloss)+"/Videos/"+"%#05d.mp4" % (count+1), "Data/"+str(gloss)+"/"+"%#05d" % (count+1) )
                    count = count+1
        if count == 0:
            print("No annotation found with this name")