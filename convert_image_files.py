from PIL import Image 
import os 

def png_to_jpg(path_to_directory):
    for file in os.listdir(path_to_directory): 
        if file.endswith(".png"):
            filepath =  path_to_directory + '/' + file
            img = Image.open(filepath) # check if we can open the file
            # if the image is None then we could not load it from disk (so skip it)
            if img is None:
                os.remove(filepath) # remove file if not readable
                continue
            #print(f"{file} ----> {img.mode}")
            # check mode of image
            if not img.mode == 'RGB':
                img = img.convert('RGB') # convert to RGB
            file_name, file_ext = os.path.splitext(file)
            img.save('{}.jpg'.format(path_to_directory + '/' + file_name))
            # delete the png file after saving it as jpg
            os.remove(filepath)