import os
import numpy as np
import json
from sklearn import preprocessing
import pandas as pd
import h5py
from signlib import misc2
from signlib import video_utils
from signlib import pose
from signlib import annotations

def create_dataset(root):
    """
    Function to create numpy array with all the files corresponding to each class and folder.
    """
    
    max_files = len(os.listdir(root))
    #print("There are %s files in the directory"%(max_files))
    
    
    empty_array = np.zeros([int(max_files/3), 7,2])
    #Get files in folders
    final_files = os.listdir(root)
    #print(final_files)
    count = 0
    for file in final_files:
        if file.endswith("json"):
            #print(file)
            json_file = root+"/"+file
            json_file_hands = misc2.get_hands_from_json(json_file)
            #json_file_hands = np.reshape(json_file_hands, (1, 14))
            
            empty_array[count] = json_file_hands
            count = count+1
    return(empty_array)


def create_dataframe_with_window(my_array, window):
#     h5f = h5py.File(h5_file,'r')
#     b = h5f['dataset_1'][:]
#     h5f.close()

    b = my_array
    #Create dataframe for storing 
    df = pd.DataFrame(columns=['Pose'])
    original_ds_shape = b.shape
    for i in range(original_ds_shape[0]):
        #print("pose is: ",b[i][j][k])
        df = df.append(pd.DataFrame({"Pose": [b[i]]}))
    #fix the index
    index = np.arange(df.shape[0])
    df.set_index(index, inplace=True)

    
    window_size = window
    final_dataframe = pd.DataFrame(columns=['Windowed_poses'])
    my_len = len(df)
    #print("length of dataframe is: %s"%(my_len))
    #print("Class "+str(c)+" has "+str(my_len)+" files in sequence: "+str(s))
            
    if my_len > window_size:
        count_windows = 0
        for u in range(my_len-window_size):
            count_windows = count_windows + 1
            #final_dataframe = final_dataframe.append(pd.DataFrame({"Class": c, "Windowed_poses": }))
            #window_array = np.asarray([new_df[new_df.Class == c][new_df.Sequence == s].Pose[u:window_size+u]])
            window_array = df.Pose[u:(window_size+u)]
            #print("Pose is: ",new_df[new_df.Class == c][new_df.Sequence == s].Pose[u])
            #print("window is: ",window_array)
            window_array = window_array.reset_index(drop=True).values.flatten()
            #window_array = np.reshape(window_array, (7,14))
            final_dataframe = final_dataframe.append(pd.DataFrame({"Windowed_poses": [window_array]}))
#           print("Class "+str(c)+" has "+str(my_len)+" files in sequence: "+str(s)+" windows in total: "+str(count_windows))
#       else:
#           print("Folders have less ")
    return(final_dataframe)


def predict_for_video(input_f, output, my_window):
#     #split to frames
#     video_utils.video_to_frames(input_f,output)
    
#     #run openpose
#     pose.run_test("E:/Testing_signlib/openpose/bin/OpenPoseDemo.exe", output)
    

    my_data = create_dataset(output)
    
    window = my_window

    mf = pd.DataFrame(columns=['Window_prediction'])

    df = create_dataframe_with_window(my_data, window)

    for n in range(df["Windowed_poses"].shape[0]):
        df["Windowed_poses"].iloc[n] = np.hstack(df["Windowed_poses"].iloc[n]).reshape((window,14))

    data = np.hstack(df["Windowed_poses"]).reshape((df["Windowed_poses"].shape[0],window,14))

    print("shape of data: ",data.shape)
    
    # load keras model
    from keras.models import load_model
    model = load_model('./keras_models/binary_classification_window_20_lstm_64_roc_best.h5')
    for sample in range(data.shape[0]):
        window_prediction = model.predict_classes(np.array(data[sample]).reshape((1,window,14)))
        mf = mf.append(pd.DataFrame({"Window_prediction": [window_prediction]}))
        
    root2 = output
    all_files = os.listdir(root2)
    pic_list = []
    for i_file in all_files:
        if i_file.endswith("rendered.png"):
            pic_list.append(i_file)

    for w in range (len(pic_list) - window):
        prediction = mf.Window_prediction.iloc[w]
        for picture in pic_list[w: window+w]:
            annotations.create_binary_overlay(picture, prediction, root2)


