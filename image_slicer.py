from PIL import Image
import os
import glob
import shutil



def crop(path, input, prefix):

    im = Image.open(input)
    imgwidth, imgheight = im.size
    horiz = 2
    vert = 3
    width = imgwidth // horiz
    height = imgheight // vert
    k = 3
    for i in range(height,imgheight,height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            a.save(os.path.join(path, 'cropped_{}_{}.jpg'.format(prefix, k)))
            k +=1


if __name__ == '__main__':
    
    max_img_count = 50

    for i in range(40):
        output_path = 'celebA_results/test_outputs/' 
        folders = os.path.join(output_path, 'test_out_{}_*'.format(i))
        #print(folders)
        folder_names = glob.glob(folders)
        #folder_names.sort()
        
        for origpath in folder_names:
            #origpath = folder_names[0]
            print(folder_names)
            print(origpath)
            path = os.path.join(origpath, 'cropped')
            filenames = glob.glob(os.path.join(origpath, '*.jpg'))
            
            try:
                shutil.rmtree(path)
            except:
                pass

            try:
                os.mkdir(path)
            except:
                pass

            for i, f in enumerate(filenames):
                if i > 50:
                    break
                splits = f.split('/')
                prefix = splits[-1].split('.')[0]

                crop(path, f, prefix)

    