import cv2
import sys
import os
import scipy
import numpy as np

class FaceCropper(object):
    CASCADE_PATH = "haarcascade_frontalface_default.xml"

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)

    def generate(self, image_path, outpath, show_result=False):
        img = cv2.imread(image_path)
        if (img is None):
            print("Can't open image file")
            return 0

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(img, 1.1, 3, minSize=(25, 25))
        if (faces is None):
            print('Failed to detect face')
            return 0

        if (show_result):
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
            cv2.imshow('img', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        facecnt = len(faces)
        #print("Detected faces: %d" % facecnt)
        i = 0
        height, width = img.shape[:2]

        for (x, y, w, h) in faces:
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            faceimg = img[ny:ny+nr, nx:nx+nr]
            lastimg = cv2.resize(faceimg, (64, 64))
            i += 1
            cv2.imwrite(outpath, lastimg)
            #print('Wrote to ' + outpath)
            break


def process_folder():
    # 5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young 
    labels = '5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young '
    labels = labels.split()
    labels.insert(0, 'fname')

    index_male = labels.index('Male')
    index_glasses = labels.index('Eyeglasses')
    index_no_beard = labels.index('No_Beard')
    index_makeup = labels.index('Heavy_Makeup')

    truth_data = {}
    truth_path = 'list_attr_celeba.txt'
    with open(truth_path, 'r') as f:
        lines = f.readlines()

        for l in range(2, len(lines)):
            splits = lines[l].split()
            #print(splits)
            temp = []
            for s in splits:
                temp.append(s == '1')
            truth_data[splits[0]] = temp

    #print(truth_data)
    #exit()
    detector = FaceCropper()

    count = 0

    active_counter = 0
    clean_counter = 0

    img_path = 'img_align_celeba'
    out_img_path_active = 'celebA_out/glasses_and_beard'
    out_img_path_clean = 'celebA_out/clean'
    out_img_path_active_test = 'celebA_out/glasses_and_beard_test'
    out_img_path_clean_test = 'celebA_out/clean_test'

    for fname in os.listdir(img_path):
        
        count += 1
        if count % 100 == 0:
            print('{}/{}'.format(count, len(truth_data)))

        full_path = os.path.join(img_path, fname)
        truth = truth_data[fname]
        
        if truth[index_makeup]:
            continue
        
        if truth[index_male] == False:
            continue
        
        if truth[index_glasses] and (truth[index_no_beard] == False):
            active_counter += 1
            #print('active')
            if active_counter % 8 == 0:
                detector.generate(full_path, os.path.join(out_img_path_active_test, fname))
            else:
                detector.generate(full_path, os.path.join(out_img_path_active, fname))

        elif not (truth[index_glasses] or (not truth[index_no_beard])):
            
            if clean_counter > active_counter * 3:
                continue
            clean_counter += 1
            #print('clean')
            if clean_counter % 8 == 0:
                detector.generate(full_path, os.path.join(out_img_path_clean_test, fname))
            else:
                detector.generate(full_path, os.path.join(out_img_path_clean, fname))


if __name__ == '__main__':
    process_folder()