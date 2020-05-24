import os, cv2, glob
import pydicom as dicom


class dataset:

    def __init__(self, dir_path):
        self.dir_path = dir_path

    def load_files(self):
        
        images = []
        masks = []
        labels = []

        for folder in os.listdir(self.dir_path):
            folder_path = os.path.join(self.dir_path, folder)
            mask_path = os.path.join(folder_path, "lung_mask")
            for img in glob.glob(folder_path+"/*.dcm"):
                # Label 
                labels.append(folder)
                # Lung image
                images.append(dicom.read_file(img))
                # Lung mask image
                mask_img_path = "lung_mask_"+str(int(img[-13:-9]))+"_"+str(int(img[-8:-4]))+".dcm"
                mask_img_path = os.path.join(mask_path, mask_img_path)
                masks.append(dicom.read_file(mask_img_path))
        
        return images, masks, labels


