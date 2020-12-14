from django.shortcuts import render
from .forms import ImageUploadForm
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

#now We have to save the file and which will be handled by our model and as it is coming in chunks from the form
#coz of encryption now we have to save it in chunks
def handle_uploaded_file(f):
     #saving the file using this function
    with open('img.jpg','wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)


# Create your views here.
def home(request):
    return render(request,'home.html')

def imageprocess(request):
    form =ImageUploadForm(request.POST,request.FILES)
    if form.is_valid():
        handle_uploaded_file(request.FILES['image'])

        base_model = ResNet50(weights='imagenet')
        img_path = 'img.jpg'
        
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = base_model.predict(x)
        print('Predicted:', decode_predictions(preds, top=3)[0])


        #giving 3 contents from result 
        html = decode_predictions(preds,top=3)[0]
        res = []
        for e in html:
            res.append((e[1],np.round(e[2]*100,2)))
        return render(request,'result.html',{'res':res})
    return render(request,'result.html')