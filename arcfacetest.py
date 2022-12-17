## ArcFace Code Experimenting for CPS843
import os
from deepface import DeepFace
from deepface.commons import distance as dst
from deepface.commons import functions
import matplotlib.pyplot as plt

models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']

model = DeepFace.ArcFace.loadModel()
target_size = model.layers[0].input_shape[1:3]

images = []
processImages = []

for image in os.listdir("images/"):
        f = os.path.join("images/", image)
        if os.path.exists:
            images.append(image)

#Preprocess image to 112x112 for ArcFace model
for image in images:
    processImages.append(DeepFace.functions.preprocess_face("images/" + image, target_size = (112, 112), detector_backend = "mtcnn"))

distances = []
def verify(img1, img2, distances):

    print("Creating embeddings \n")
    img1_embedding = model.predict(img1)[0]
    img2_embedding = model.predict(img2)[0]

    print("Finding Distance\n")
    distance = dst.findCosineDistance(img1_embedding, img2_embedding)
    #Threshold based on internal model parameters
    threshold = 0.6871912959056619
    distances.append(distance)
    
    fig = plt.figure()

    if distance <= threshold:
        plt.suptitle('Image Shows Same Person, Distance is: \n')
        plt.title(round(distance, 10))
        plt.axis('off')
        print("Images are the same person")
    else:
        plt.suptitle('Image Shows Different People, Distance is: \n')
        plt.title(round(distance, 10))
        plt.axis('off')
        print("Images are different people")
    
    print("Distance is ",round(distance, 10))
    print("Threshold is ",round(threshold, 10))

    ax1 = fig.add_subplot(1,2,1)
    plt.axis('off')
    plt.imshow(img1[0][:,:,::-1])
    
    
    ax2 = fig.add_subplot(1,2,2)
    plt.axis('off')
    plt.imshow(img2[0][:,:,::-1])
    
    plt.show()

for index, image in enumerate(processImages):
    if (index < len(processImages) - 1):
        verify(image, processImages[index+1], distances)
