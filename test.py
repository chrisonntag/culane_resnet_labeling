from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import preprocessing as pp
from train import fbeta
from keras import backend


model = load_model('culane_model', custom_objects={"fbeta": fbeta})

img = pp.prepare_img('unseen_labeled/dazzlelight_shadow_curve.jpg')
pred = model.predict(img)

print(pred[0])
print(pp.pred_tags(pred[0]))
