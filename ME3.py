# Machine Exercise # 3
from ME2 import distort

def generate_dataset(
    input_path,
    output_path
):
    import cv2
    import os
    import csv
    import numpy as np

    og_img = list()
    og_img_path = list()
    og_img_dir = input_path + '/images/images'
    og_img_list = os.listdir(og_img_dir)

    input_path = output_path + '/original'
    output_path = output_path + '/warped'

    try:
        os.mkdir('pokemon_dataset')
    except FileExistsError:
        pass

    try:
        os.mkdir(output_path)
    except FileExistsError:
        pass

    try:
        os.mkdir(input_path)
    except FileExistsError:
        pass


    for img in og_img_list:
        path = og_img_dir + '/' + img
        og_img_path.append(path)

    img_og = []
    img_warped = []
    pokename_arr = []
    # create warped images
    for img in og_img_path:
        pokename = img[38:]
        og_img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        #cv2.imwrite(input_path + pokename, og_img)

        pokename = pokename.replace('.png', '_warped.png')
        pokename = pokename.replace('.jpg', '_warped.png')
        # warp it baby
        patch = [
            [10, 10],
            [10, 30],
            [30, 30],
            [30, 10]
        ]
        distorted_img = distort(
            img,
            os.getcwd() + '/' + output_path + pokename,
            patch
        )
        og_filename = input_path + pokename
        np.savez_compressed(
            og_filename,
            img=og_img
        )

        warped_filename = os.getcwd() + '/' + output_path + pokename
        np.savez_compressed(
            warped_filename,
            img=distorted_img
        )
        print(warped_filename+ '.npz')
        #data = np.load(warped_filename+ '.npz', allow_pickle=True)
        #print(data['img'])


generate_dataset(
    'pokemon-images-and-types',
    'pokemon_dataset'
)