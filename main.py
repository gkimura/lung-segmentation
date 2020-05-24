import numpy as np, cv2
from dataset import dataset

def convert_to_hounsfield_units(files):

    images = np.stack([f.pixel_array.astype(np.int16) for f in files])
    images[images==-2000] = 0
    intercept  = files[0].RescaleIntercept
    slope = files[0].RescaleSlope
    
    images *= np.int16(slope)
    images += np.int16(intercept)
    images = np.array(images, dtype=np.int16)    
    
    return images 

if __name__ == "__main__":

    data = dataset("dataset/")
    print("--Loading files")
    images, masks, labels = data.load_files()

    print("--Converting to Hounsfield units")
    hu_images = convert_to_hounsfield_units(images)
    hu_masks = convert_to_hounsfield_units(masks)

