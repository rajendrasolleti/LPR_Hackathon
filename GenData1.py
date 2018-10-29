import os
import numpy as np
import cv2

input= r'C:\Users\v-rasoll\Documents\EnglishFnt\English\Fnt'
directories = os.listdir(input)
f = open(r'C:\hackathon\project\ascii.csv.txt', encoding = 'utf-16')
content = f.read().split('\n')
intClassifications = []
npaFlattenedImages =  np.empty((0, 1800))
for i in range(len(content)):
    print(content[i].split('\t')[0])
    char_ascii_code = content[i].split('\t')[1]
    char_location_paths = os.listdir(input + "\\" + directories[i])
    j=0
    for each_character_image in char_location_paths:
        j= j+1
        if j<200 :
            intClassifications.append(char_ascii_code)
            char_image_path = input + "\\" + directories[i] + "\\" + each_character_image
            char_image = cv2.imread(char_image_path)

            char_image_resized = cv2.resize(char_image, (20,30))
            npaFlattenedImage = char_image_resized.reshape((1, 1800))
            npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)
            cv2.destroyAllWindows()
fltClassifications = np.array(intClassifications, np.float32)
npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))

np.savetxt(r"C:\hackathon\project\classifications.txt", npaClassifications)           # write flattened images to file
np.savetxt(r"C:\hackathon\project\flattened_images.txt", npaFlattenedImages)