import numpy as np

def convert_string_to_numpy_array(string, cam_width=640, cam_height=480):
    split_string = string.split(',')[:-1]
    # print(len(split_string))
    numpy_array = np.array([(float(i) - cam_width / 2) / 100 if index % 3 == 0 
                            else (float(i) - cam_height / 2) / 100 if index % 3 == 1 
                            else (float(i) / 100) for index, i in enumerate(split_string)])
    
    # numpy_array = np.array([(float(i) - cam_width / 2) / 100 if index % 3 == 0 
    #                         else (float(i) - cam_height / 2) / 100 if index % 3 == 1 
    #                         else 1 for index, i in enumerate(split_string)])
    # numpy_array = np.array([float(i) / 100 if index % 3 != 2 else 0 for index, i in enumerate(split_string)])
    # print(f'STN: {numpy_array[:3]}')
    return numpy_array.reshape(-1, 3 )