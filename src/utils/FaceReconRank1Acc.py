import os
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display
from tqdm import tqdm
folder_path = 'data/test/no_glasses/'
files_no_glass = os.listdir(folder_path)
#
# 
#
# folder_path2 = 'data/test/glasses/' # replace this dir with removed glasses folder, keep names the same when removing glasses!!!!
folder_path2 = '/home/josh/remglass/data/all_results/' # replace this dir with removed glasses folder, keep names the same when removing glasses!!!!

files_glass = os.listdir(folder_path2)

print(len(files_glass), len(files_no_glass))
known_face_encodings=[]
known_face_names = []
valid_face = []


for i in tqdm(files_no_glass):
    try:
        noglass = face_recognition.load_image_file(folder_path+i)
        noglass = face_recognition.face_encodings(noglass)[0]
        known_face_encodings.append(noglass)
        known_face_names.append(i[:15])
        valid_face.append(i)
    except:
        pass
    
print('Learned encoding for', len(known_face_encodings), 'images.')
correct = 0
wrong=0
total=0
TP=0
TN=0
FP=0
FN=0
import random
for i in tqdm(valid_face):
    idx_1 = random.randint(0,len(valid_face))
    idx_2 =  random.randint(0,len(valid_face))
    for idx, j in enumerate(files_glass):
        if i[:15]==j[:15]:

            unknown_image = face_recognition.load_image_file(folder_path2+j)

            # Find all the faces and face encodings in the unknown image
            face_locations = face_recognition.face_locations(unknown_image)
            face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
            for _, face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                if name == i[:15]:
                    TP+=1
                else:
                    FN+=1
                total+=1
        if idx==idx_2 or idx==idx_1:

            unknown_image = face_recognition.load_image_file(folder_path2+j)

            # Find all the faces and face encodings in the unknown image
            face_locations = face_recognition.face_locations(unknown_image)
            face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
            for _, face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                if name == i[:15]:
                    FP+=1
                else:
                    TN+=1   
                total+=1
Acc = (TP+TN)/(TP+TN+FN+FP)
print('Acc:',Acc)
Recall = TP/(TP+FN)
print('Recall:',Recall)
Precision = TP/(TP+FP)
print('Precision:',Precision)
F1 = (2*Precision*Recall)/(Precision+Recall)
print('F1:', F1)
##### itterate whole dataset (cost more than 4h)
quit()
TP = {i: 0 for i in valid_face}
TN = {i: 0 for i in valid_face}
FP = {i: 0 for i in valid_face}
FN = {i: 0 for i in valid_face}
total = {i: 0 for i in valid_face}
for i in tqdm(valid_face):
    for j in files_glass[:1]:
        if True:
            unknown_image = face_recognition.load_image_file(folder_path2+j)

            # Find all the faces and face encodings in the unknown image
            face_locations = face_recognition.face_locations(unknown_image)
            face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
            for _, face_encoding in zip(face_locations, face_encodings):
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                if name == i[:15]:  
                    TP[i] += 1
                else:
                    if name == "Unknown":
                        FN[i] += 1
                    else:
                        FP[i] += 1
                total[i] += 1

precision = {i: TP[i] / (TP[i] + FP[i]+1e-9) for i in valid_face}
recall = {i: TP[i] / (TP[i] + FN[i]+1e-9) for i in valid_face}

macro_precision = sum(precision.values()) / len(precision)
macro_recall = sum(recall.values()) / len(recall)
macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall+1e-9)

micro_TP = sum(TP.values())
micro_FP = sum(FP.values())
micro_FN = sum(FN.values())
micro_precision = micro_TP / (micro_TP + micro_FP+1e-9)
micro_recall = micro_TP / (micro_TP + micro_FN+1e-9)
micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall+1e-9)

print("Macro-averaged precision:", macro_precision)
print("Macro-averaged recall:", macro_recall)
print("Macro-averaged F1 score:", macro_f1)
print("Micro-averaged precision:", micro_precision)
print("Micro-averaged recall:", micro_recall)
print("Micro-averaged F1 score:", micro_f1)
