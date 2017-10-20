# We import some tools from the Keras API.
# A CNN model named ResNet50 in addition to some related precessing tools
# We import also numpy, a very important package for scientific computing
# in addition to openCV (cv2)
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import cv2

# Here, we define and load our model ResNet50. It is great in Keras,
# you don't even need to write the model etc. etc. you just call it, everything is done in Keras
# We also load pre-trained weights. These weights have been trained on a huge dataset called 'imageNet'
# so, you don't even make any training for now, just load and play!
model = ResNet50(weights='imagenet')

# Define the path to your input image
img_path = 'data/cat.jpg'

# We process first our data, and this depends on the model we are using
# here for example we need always to resize our image to 224 by 224
# there is this function preprocess_input that makes the necessary transformations for our image
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Now we classify our image using the 'predict' function.
# The output is actually a vector of 1 by 1000 because our model was trained for 1000 classes
# this vector contains probabilities regarding every class, so the highest probability should give us
# the correct prediction. Well we are not sure, so we generally check the highest 3 values.
# The function decode_predictions do the job for us, we don't need to write all the 1000 classes
# and then try to interpret the output values, it does everything for us.
# You can show the top 1, top 3, top 5... etc.

preds = model.predict(x)
decoded = np.array(decode_predictions(preds, top=3))[0]
print('Predicted:', decoded)

# Here, let's use openCV to show our image and add the label and the highest probability on it
orig = cv2.imread(img_path)
cv2.putText(orig, "Label: {}, {:.2f}%".format(decoded[0][1], float(decoded[0][2])*100),
						(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)

# PS: if you don't make cv2.waitKey(0), the picture will disapear quickly, don't forget to use it!

# Tadaaaa! we are done! enjoy!
