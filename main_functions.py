import args


def step1_get_index_from_args():
    import args
    index, sAddr, sPort, debug = args.get_args()
    return index, sAddr, sPort, debug


def step2_get_webcam_list():
    import modules.webcam_list as webcam_list
    webcam_lst = webcam_list.get_all_webcams()
    webcam_lst = webcam_list.get_available_webcams(webcam_lst)
    return webcam_lst


def step3_get_webcam_index(index, webcam_lst):
    import modules.webcam_list as webcam_list
    webcam_lst = webcam_list.get_webcam_index(index, webcam_lst)
    return webcam_lst


def step4_is_indexed_webcam_not_available(webcam_lst):
    return len(webcam_lst) == 0