import numpy as np, cv2
from dataset import dataset
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import disk, binary_dilation, binary_erosion
from skimage.morphology import binary_closing, remove_small_objects
from skimage.segmentation import clear_border
from skimage import measure, filters

def display(img, mask, label):
    fig, axs = plt.subplots(1,2)
    fig.suptitle("Lung Segmentation Class "+label)
    axs[0].imshow(mask, cmap='gray')
    axs[0].set_title("Mask")
    axs[1].imshow(img, cmap='gray')
    axs[1].set_title("Image")
    plt.show()

def dice_coefficient(img, mask):
    h, w = img.shape
    tp=fp=fn=0

    for i in range(h):
        for j in range(w):
            if img[i,j]==True and mask[i,j]==255:
                tp+=1
            elif img[i,j]==True and mask[i,j]==0:
                fp+=1
            elif img[i,j]==False and mask[i,j]==255:
                fn+=1

    dc = (2.0*tp)/(2.0*tp+fp+fn)
    return dc

def convert_to_hounsfield_units(files):

    images = np.stack([f.pixel_array.astype(np.int16) for f in files])
    images[images==-2000] = 0
    intercept  = files[0].RescaleIntercept
    slope = files[0].RescaleSlope
    
    images *= np.int16(slope)
    images += np.int16(intercept)
    images = np.array(images, dtype=np.int16)    
    
    return images 

def threshold_selection(image):
    th = (np.amax(image) + np.amin(image))/2
    th_ant = 0
    while(th != th_ant):
        b = image[image<th]
        n = image[image>th]
        th_ant = th
        th = (np.mean(b)+np.mean(n))/2
    return int(th)

if __name__ == "__main__":

    data = dataset("dataset/")
    print("-- Loading files")
    images, masks, labels = data.load_files()

    print("-- Converting to Hounsfield units")
    hu_images = convert_to_hounsfield_units(images)
    hu_masks = convert_to_hounsfield_units(masks)

    print("-- Processing images")
    scores = []
    for img, mask, label in zip(hu_images, hu_masks, labels):

        th = threshold_selection(img)
        img = (img < th -30).astype(int)
        
        # Remove background
        img = clear_border(img, bgval=0)

        img = binary_dilation(img,disk(2))
        img = binary_erosion(img, disk(1))

        # Remove small objects
        img = remove_small_objects(img, 50)
        img = binary_closing(img, disk(5))

        # Fill holes inside the lungs
        edges = filters.roberts(img)
        img = ndi.binary_fill_holes(img)
        img = binary_erosion(img,disk(1))
        img = remove_small_objects(img,100)

        dc = dice_coefficient(img, mask)
        scores.append(dc)
        print("   Label: "+label+"  DC:",np.round(dc*100,2))
        display(img,mask,label)

    print("-- Mean Dice coefficient: ", np.round(np.mean(scores)*100,2))


