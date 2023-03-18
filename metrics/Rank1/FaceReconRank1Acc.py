import os
import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display
from tqdm import tqdm
folder_path = 'data/test/no_glasses/'
files_no_glass = os.listdir(folder_path)
# replace this dir with removed glasses folder, keep names the same when remove glasses
folder_path2 = 'data/test/glasses/'
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
total=0
for i in tqdm(valid_face):
    for j in files_glass:
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
                    correct+=1
                total+=1
rank1acc = correct/total
print(rank1acc)