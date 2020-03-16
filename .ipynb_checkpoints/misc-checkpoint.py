import numpy as np
import json
from sklearn import preprocessing
import os
import numpy as np
import pandas as pd
import h5py

def get_hands_from_json(json_file_path):
    """
    function to open a json file and return the x,y positions of the hands.
    Provide the path to the json file
    """
    with open(json_file_path) as f:
        loaded_json = json.load(f)
        raw_coords = loaded_json["people"][0]["pose_keypoints_2d"]
        #remove confidence values
        raw_coords = np.delete(raw_coords, np.arange(2, len(raw_coords), 3))
        raw_coords = np.reshape(raw_coords, (18,2))

        raw_coords = np.array(raw_coords)
        hands_array = np.array([raw_coords[0],raw_coords[2],raw_coords[3],raw_coords[4],raw_coords[5],raw_coords[6],raw_coords[7]])
        hands_array = preprocessing.scale(hands_array)
        return(hands_array) 

def get_list_of_directories(my_path):
    my_list_of_directories = []
    not_last = []
    root = my_path
    first_dir= os.listdir(root) # get all files' and folders' names in the current directory
    #print("First directory is: ",first_dir)

    for direc in first_dir:
        second_dir = root+"/"+direc
        second_dir_dir = os.listdir(second_dir)
        #print("second directory is: ",second_dir_dir)
        for final_dir in second_dir_dir:
            if (final_dir != "Videos"):
                directory_to_save = second_dir+"/"+final_dir
                final_directory = str(directory_to_save)
                not_last.append(final_dir)
        
        my_list_of_directories.append([[direc], [not_last]])
        df = pd.DataFrame.from_records(my_list_of_directories)
        not_last = []
    return(df)

def get_files_and_folders(root):
    dir_list = next(os.walk(root))[1]
    max_n_of_folders = 0
    max_n_of_files = 0
    for first_index in dir_list:
        next_dir = next(os.walk(root+"/"+first_index))[1]
        if len(next_dir) > max_n_of_folders:
            max_n_of_folders = len(next_dir)
        #print("The directory: "+first_index+" contains "+str(len(next_dir))+" folders")
        for folder in next_dir:
            final_files = os.listdir(root+"/"+first_index+"/"+folder)
            if len(final_files) > max_n_of_files:
                max_n_of_files = len(final_files)

            #print("The directory: "+first_index+" has a folder "+str(folder)+" which contains no of files: "+str(len(final_files)))
    return(len(dir_list), max_n_of_folders, max_n_of_files)

def create_dataset(root):
    """
    Function to create numpy array with all the files corresponding to each class and folder.
    """
    classes, max_folders, max_files = get_files_and_folders(root)
    
    empty_array = np.zeros([classes, max_folders, int(max_files/3), 7,2])
    
    dir_list = next(os.walk(root))[1]
    for first_index in dir_list:
        #print("first index ",dir_list.index(first_index))
        next_dir = next(os.walk(root+"/"+first_index))[1]
        #print("The directory: "+first_index+" contains "+str(len(next_dir))+" folders")
        for folder in next_dir:
            final_files = os.listdir(root+"/"+first_index+"/"+folder)
            #print("folder is: ",next_dir.index(folder))
            count = 0
            for file in final_files:
                if file.endswith("json"):
                    json_file = root+"/"+first_index+"/"+folder+"/"+file
                    json_file_hands = get_hands_from_json(json_file)
                    #json_file_hands = np.reshape(json_file_hands, (1, 14))
                    json_file_hands = np.reshape(json_file_hands, (7, 2))
                    empty_array[dir_list.index(first_index)][next_dir.index(folder)][count] = json_file_hands
                    count = count+1
    return(empty_array)

def create_dataframe_with_window(h5_file, window):
    h5f = h5py.File(h5_file,'r')
    b = h5f['dataset_1'][:]
    h5f.close()
    #Create dataframe for storing 
    df = pd.DataFrame(columns=['Class', 'Sequence', 'Pose'])
    original_ds_shape = b.shape
    for i in range(original_ds_shape[0]):
        for j in range(original_ds_shape[1]):
            for k in range(original_ds_shape[2]):
                df = df.append(pd.DataFrame({"Class": i, "Sequence": j, "Pose": [b[i][j][k]]}))
    #fix the index
    index = np.arange(df.shape[0])
    df.set_index(index, inplace=True)
    
    # Remove Os from dataframe

    zeros_array = np.zeros((7,2))
    count = []
    for z in range(df.shape[0]):
        if((zeros_array == df["Pose"].iloc[z]).all()):
            #df.drop(df.index[[z]])
            count.append(z)
    new_df = df.drop(df.index[count])
    new_df.reset_index(drop=True)
    new_df["Pose"].values.reshape(302,14)
    # Reshape the data to create sequences of wwindow size
    window_size = window
    final_dataframe = pd.DataFrame(columns=['Windowed_poses', 'Class'])
    for s in new_df.Sequence.unique():
        for c in new_df.Class.unique():
            my_len = len(new_df[new_df.Class == c][new_df.Sequence == s])
            
            #print("Class "+str(c)+" has "+str(my_len)+" files in sequence: "+str(s))
            
            if my_len > window_size:
                for u in range(my_len-window_size):
                    #final_dataframe = final_dataframe.append(pd.DataFrame({"Class": c, "Windowed_poses": }))
                    #window_array = np.asarray([new_df[new_df.Class == c][new_df.Sequence == s].Pose[u:window_size+u]])
                    window_array = new_df[new_df.Class == c][new_df.Sequence == s].Pose[u:window_size+u].values.flatten()
                    print("Pose is: ",new_df[new_df.Class == c][new_df.Sequence == s].Pose[u])
                    print("window is: ",window_array)
                    window_array = window_array.flatten()
                    final_dataframe = final_dataframe.append(pd.DataFrame({"Windowed_poses": window_array, "Class": c}))
    return(final_dataframe)