from django.shortcuts import render, redirect
import os
import numpy as np
import cv2
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
try:
    import face_recognition
except ImportError:
    face_recognition = None
import time
import sys
import json
import glob
import copy
import shutil
from PIL import Image as pImage
import time
from django.conf import settings
from .forms import VideoUploadForm

try:
    import torch
    import torchvision
    from torchvision import transforms, models
    from torch.utils.data import DataLoader
    from torch.utils.data.dataset import Dataset
    from torch.autograd import Variable
    from torch import nn
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    torchvision = None
    transforms = None
    models = None
    DataLoader = None
    Dataset = object
    Variable = None
    nn = None
    _TORCH_AVAILABLE = False

index_template_name = 'index.html'
predict_template_name = 'predict.html'
about_template_name = "about.html"

im_size = 112
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

if _TORCH_AVAILABLE:
    sm = nn.Softmax()
    inv_normalize =  transforms.Normalize(mean=-1*np.divide(mean,std),std=np.divide([1,1,1],std))
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])
else:
    sm = None
    inv_normalize = None
    device = 'cpu'
    train_transforms = None

if _TORCH_AVAILABLE:

    class Model(nn.Module):

        def __init__(self, num_classes,latent_dim= 2048, lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
            super(Model, self).__init__()
            model = models.resnext50_32x4d(pretrained = True)
            self.model = nn.Sequential(*list(model.children())[:-2])
            self.lstm = nn.LSTM(latent_dim,hidden_dim, lstm_layers,  bidirectional)
            self.relu = nn.LeakyReLU()
            self.dp = nn.Dropout(0.4)
            self.linear1 = nn.Linear(2048,num_classes)
            self.avgpool = nn.AdaptiveAvgPool2d(1)

        def forward(self, x):
            batch_size,seq_length, c, h, w = x.shape
            x = x.view(batch_size * seq_length, c, h, w)
            fmap = self.model(x)
            x = self.avgpool(fmap)
            x = x.view(batch_size,seq_length,2048)
            x_lstm,_ = self.lstm(x,None)
            return fmap,self.dp(self.linear1(x_lstm[:,-1,:]))


    class validation_dataset(Dataset):
        def __init__(self,video_names,sequence_length=60,transform = None):
            self.video_names = video_names
            self.transform = transform
            self.count = sequence_length

        def __len__(self):
            return len(self.video_names)

        def __getitem__(self,idx):
            video_path = self.video_names[idx]
            frames = []
            a = int(100/self.count)
            first_frame = np.random.randint(0,a)
            for i,frame in enumerate(self.frame_extract(video_path)):
                #if(i % a == first_frame):
                faces = face_recognition.face_locations(frame) if face_recognition else []
                try:
                    top,right,bottom,left = faces[0]
                    frame = frame[top:bottom,left:right,:]
                except:
                    pass
                frames.append(self.transform(frame))
                if(len(frames) == self.count):
                    break
            """
            for i,frame in enumerate(self.frame_extract(video_path)):
                if(i % a == first_frame):
                    frames.append(self.transform(frame))
            """        
            # if(len(frames)<self.count):
            #   for i in range(self.count-len(frames)):
            #         frames.append(self.transform(frame))
            #print("no of frames", self.count)
            frames = torch.stack(frames)
            frames = frames[:self.count]
            return frames.unsqueeze(0)
        
        def frame_extract(self,path):
          vidObj = cv2.VideoCapture(path) 
          success = 1
          while success:
              success, image = vidObj.read()
              if success:
                  yield image
else:
    class Model:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for model inference")

    class validation_dataset:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for video preprocessing")

def im_convert(tensor, video_file_name):
    """ Display a tensor as an image. """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for image conversion")
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1,2,0)
    image = image.clip(0, 1)
    # This image is not used
    # cv2.imwrite(os.path.join(settings.PROJECT_DIR, 'uploaded_images', video_file_name+'_convert_2.png'),image*255)
    return image

def im_plot(tensor):
    if plt is None:
        return
    image = tensor.cpu().numpy().transpose(1,2,0)
    b,g,r = cv2.split(image)
    image = cv2.merge((r,g,b))
    image = image*[0.22803, 0.22145, 0.216989] +  [0.43216, 0.394666, 0.37645]
    image = image*255.0
    plt.imshow(image.astype('uint8'))
    plt.show()


def predict(model,img,inference_device,path = './', video_file_name=""):
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for prediction")
    fmap,logits = model(img.to(inference_device))
    img = im_convert(img[:,-1,:,:,:], video_file_name)
    params = list(model.parameters())
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    logits = sm(logits)
    _,prediction = torch.max(logits,1)
    confidence = logits[:,int(prediction.item())].item()*100
    print('confidence of prediction:',logits[:,int(prediction.item())].item()*100)
    return [int(prediction.item()),confidence]

def plot_heat_map(i, model, img, path = './', video_file_name=''):
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for heat map generation")
    fmap,logits = model(img.to(device))
    params = list(model.parameters())
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    logits = sm(logits)
    _,prediction = torch.max(logits,1)
    idx = np.argmax(logits.detach().cpu().numpy())
    bz, nc, h, w = fmap.shape
    #out = np.dot(fmap[-1].detach().cpu().numpy().reshape((nc, h*w)).T,weight_softmax[idx,:].T)
    out = np.dot(fmap[i].detach().cpu().numpy().reshape((nc, h*w)).T,weight_softmax[idx,:].T)
    predict = out.reshape(h,w)
    predict = predict - np.min(predict)
    predict_img = predict / np.max(predict)
    predict_img = np.uint8(255*predict_img)
    out = cv2.resize(predict_img, (im_size,im_size))
    heatmap = cv2.applyColorMap(out, cv2.COLORMAP_JET)
    img = im_convert(img[:,-1,:,:,:], video_file_name)
    result = heatmap * 0.5 + img*0.8*255
    # Saving heatmap - Start
    heatmap_name = video_file_name+"_heatmap_"+str(i)+".png"
    image_name = os.path.join(settings.PROJECT_DIR, 'uploaded_images', heatmap_name)
    cv2.imwrite(image_name,result)
    # Saving heatmap - End
    result1 = heatmap * 0.5/255 + img*0.8
    r,g,b = cv2.split(result1)
    result1 = cv2.merge((r,g,b))
    return image_name

def load_checkpoint_to_model(model, model_path, map_location):
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for loading checkpoints")
    checkpoint = torch.load(model_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_key = key.replace('module.', '') if key.startswith('module.') else key
        cleaned_state_dict[cleaned_key] = value

    model.load_state_dict(cleaned_state_dict)
    return model

def parse_model_metadata(model_filename):
    name_without_extension = os.path.splitext(model_filename)[0]
    parts = name_without_extension.split("_")
    if len(parts) < 4:
        return None
    try:
        accuracy = int(parts[1])
        sequence_length = int(parts[3])
    except ValueError:
        return None

    display_name = f"{model_filename} ({sequence_length} frames, {accuracy}% acc)"
    return {
        "filename": model_filename,
        "accuracy": accuracy,
        "sequence_length": sequence_length,
        "display_name": display_name,
    }

def get_all_available_models():
    list_models = glob.glob(os.path.join(settings.PROJECT_DIR, "models", "*.pt"))
    parsed_models = []
    for model_path in list_models:
        model_filename = os.path.basename(model_path)
        metadata = parse_model_metadata(model_filename)
        if metadata is None:
            continue
        metadata["path"] = model_path
        parsed_models.append(metadata)

    parsed_models.sort(key=lambda item: (item["sequence_length"], -item["accuracy"], item["filename"]))
    return parsed_models

def get_model_choices():
    return [(model_info["filename"], model_info["display_name"]) for model_info in get_all_available_models()]

# Model Selection
def get_accurate_model(sequence_length):
    model_name = []
    sequence_model = []
    final_model = ""
    list_models = glob.glob(os.path.join(settings.PROJECT_DIR, "models", "*.pt"))

    for model_path in list_models:
        model_name.append(os.path.basename(model_path))

    for model_filename in model_name:
        try:
            seq = model_filename.split("_")[3]
            if int(seq) == sequence_length:
                sequence_model.append(model_filename)
        except IndexError:
            pass  # Handle cases where the filename format doesn't match expected

    if len(sequence_model) > 1:
        accuracy = []
        for filename in sequence_model:
            acc = filename.split("_")[1]
            accuracy.append(acc)  # Convert accuracy to float for proper comparison
        max_index = accuracy.index(max(accuracy))
        final_model = os.path.join(settings.PROJECT_DIR, "models", sequence_model[max_index])
    elif len(sequence_model) == 1:
        final_model = os.path.join(settings.PROJECT_DIR, "models", sequence_model[0])
    else:
        print("No model found for the specified sequence length.")  # Handle no models found case

    return final_model

ALLOWED_VIDEO_EXTENSIONS = set(['mp4','gif','webm','avi','3gp','wmv','flv','mkv'])

def allowed_video_file(filename):
    #print("filename" ,filename.rsplit('.',1)[1].lower())
    if (filename.rsplit('.',1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS):
        return True
    else: 
        return False
def index(request):
    model_choices = get_model_choices()
    if request.method == 'GET':
        video_upload_form = VideoUploadForm(model_choices=model_choices)
        if 'file_name' in request.session:
            del request.session['file_name']
        if 'preprocessed_images' in request.session:
            del request.session['preprocessed_images']
        if 'faces_cropped_images' in request.session:
            del request.session['faces_cropped_images']
        if 'selected_model_path' in request.session:
            del request.session['selected_model_path']
        if 'selected_model_name' in request.session:
            del request.session['selected_model_name']
        if 'sequence_length' in request.session:
            del request.session['sequence_length']
        return render(request, index_template_name, {"form": video_upload_form})
    else:
        video_upload_form = VideoUploadForm(request.POST, request.FILES, model_choices=model_choices)
        if video_upload_form.is_valid():
            if len(model_choices) == 0:
                video_upload_form.add_error(None, "No model files found in models folder")
                return render(request, index_template_name, {"form": video_upload_form})

            video_file = video_upload_form.cleaned_data['upload_video_file']
            video_file_ext = video_file.name.split('.')[-1]
            selected_model_name = video_upload_form.cleaned_data['model_name']
            selected_model_metadata = parse_model_metadata(selected_model_name)
            if selected_model_metadata is None:
                video_upload_form.add_error("model_name", "Selected model name is invalid")
                return render(request, index_template_name, {"form": video_upload_form})

            sequence_length = selected_model_metadata['sequence_length']
            selected_model_path = os.path.join(settings.PROJECT_DIR, "models", selected_model_name)
            if not os.path.exists(selected_model_path):
                video_upload_form.add_error("model_name", "Selected model file was not found")
                return render(request, index_template_name, {"form": video_upload_form})

            video_content_type = video_file.content_type.split('/')[0]
            if video_content_type in settings.CONTENT_TYPES:
                if video_file.size > int(settings.MAX_UPLOAD_SIZE):
                    video_upload_form.add_error("upload_video_file", "Maximum file size 100 MB")
                    return render(request, index_template_name, {"form": video_upload_form})

            if sequence_length <= 0:
                video_upload_form.add_error("sequence_length", "Sequence Length must be greater than 0")
                return render(request, index_template_name, {"form": video_upload_form})
            
            if allowed_video_file(video_file.name) == False:
                video_upload_form.add_error("upload_video_file","Only video files are allowed ")
                return render(request, index_template_name, {"form": video_upload_form})
            
            saved_video_file = 'uploaded_file_'+str(int(time.time()))+"."+video_file_ext
            if settings.DEBUG:
                with open(os.path.join(settings.PROJECT_DIR, 'uploaded_videos', saved_video_file), 'wb') as vFile:
                    shutil.copyfileobj(video_file, vFile)
                request.session['file_name'] = os.path.join(settings.PROJECT_DIR, 'uploaded_videos', saved_video_file)
            else:
                with open(os.path.join(settings.PROJECT_DIR, 'uploaded_videos','app','uploaded_videos', saved_video_file), 'wb') as vFile:
                    shutil.copyfileobj(video_file, vFile)
                request.session['file_name'] = os.path.join(settings.PROJECT_DIR, 'uploaded_videos','app','uploaded_videos', saved_video_file)
            request.session['sequence_length'] = sequence_length
            request.session['selected_model_path'] = selected_model_path
            request.session['selected_model_name'] = selected_model_name
            return redirect('ml_app:predict')
        else:
            return render(request, index_template_name, {"form": video_upload_form})

def predict_page(request):
    if request.method == "GET":
        # Redirect to 'home' if 'file_name' is not in session
        if 'file_name' not in request.session:
            return redirect("ml_app:home")
        if 'file_name' in request.session:
            video_file = request.session['file_name']
        if 'sequence_length' in request.session:
            sequence_length = request.session['sequence_length']
        else:
            return redirect("ml_app:home")

        selected_model_name = request.session.get('selected_model_name', '')
        selected_model_path = request.session.get('selected_model_path', '')

        path_to_videos = [video_file]
        video_file_name = os.path.basename(video_file)
        video_file_name_only = os.path.splitext(video_file_name)[0]
        # Production environment adjustments
        if not settings.DEBUG:
            production_video_name = os.path.join('/home/app/staticfiles/', video_file_name.split('/')[3])
            print("Production file name", production_video_name)
        else:
            production_video_name = video_file_name

        if not _TORCH_AVAILABLE:
            return render(request, predict_template_name, {'prediction_error': 'PyTorch is not available in this environment.'})

        # Load validation dataset
        video_dataset = validation_dataset(path_to_videos, sequence_length=sequence_length, transform=train_transforms)

        # Load model
        inference_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if selected_model_path and os.path.exists(selected_model_path):
            path_to_model = selected_model_path
        else:
            path_to_model = get_accurate_model(sequence_length)

        if not path_to_model or not os.path.exists(path_to_model):
            print(f"No model available for sequence length {sequence_length} in models directory")
            return render(request, predict_template_name, {'prediction_error': f'No model found for sequence length {sequence_length}'})

        try:
            model = Model(2).to(inference_device)
            model = load_checkpoint_to_model(model, path_to_model, inference_device)
        except RuntimeError as runtime_error:
            if "out of memory" in str(runtime_error).lower() and inference_device.type == 'cuda':
                print("CUDA out of memory during model load. Falling back to CPU inference.")
                torch.cuda.empty_cache()
                inference_device = torch.device('cpu')
                model = Model(2).to(inference_device)
                model = load_checkpoint_to_model(model, path_to_model, inference_device)
            else:
                return render(request, predict_template_name, {'prediction_error': f'Model loading failed: {runtime_error}'})
        model.eval()
        start_time = time.time()
        # Display preprocessing images
        print("<=== | Started Videos Splitting | ===>")
        preprocessed_images = []
        faces_cropped_images = []
        cap = cv2.VideoCapture(video_file)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break
        cap.release()

        print(f"Number of frames: {len(frames)}")
        # Process each frame for preprocessing and face cropping
        padding = 40
        faces_found = 0
        for i in range(sequence_length):
            if i >= len(frames):
                break
            frame = frames[i]

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Save preprocessed image
            image_name = f"{video_file_name_only}_preprocessed_{i+1}.png"
            image_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name)
            img_rgb = pImage.fromarray(rgb_frame, 'RGB')
            img_rgb.save(image_path)
            preprocessed_images.append(image_name)

            # Face detection and cropping
            if face_recognition is not None:
                face_locations = face_recognition.face_locations(rgb_frame)
                if len(face_locations) == 0:
                    continue
                top, right, bottom, left = face_locations[0]
                frame_face = frame[top - padding:bottom + padding, left - padding:right + padding]
            else:
                frame_face = frame

            # Convert cropped face image to RGB and save
            rgb_face = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)
            img_face_rgb = pImage.fromarray(rgb_face, 'RGB')
            image_name = f"{video_file_name_only}_cropped_faces_{i+1}.png"
            image_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name)
            img_face_rgb.save(image_path)
            faces_found += 1
            faces_cropped_images.append(image_name)

        print("<=== | Videos Splitting and Face Cropping Done | ===>")
        print("--- %s seconds ---" % (time.time() - start_time))

        # No face detected
        if faces_found == 0:
            return render(request, 'predict_template_name.html', {"no_faces": True})

        # Perform prediction
        try:
            heatmap_images = []
            output = ""
            confidence = 0.0

            for i in range(len(path_to_videos)):
                print("<=== | Started Prediction | ===>")
                try:
                    prediction = predict(model, video_dataset[i], inference_device, './', video_file_name_only)
                except RuntimeError as runtime_error:
                    if "out of memory" in str(runtime_error).lower() and inference_device.type == 'cuda':
                        print("CUDA out of memory. Falling back to CPU inference.")
                        torch.cuda.empty_cache()
                        inference_device = torch.device('cpu')
                        model = model.to(inference_device)
                        prediction = predict(model, video_dataset[i], inference_device, './', video_file_name_only)
                    else:
                        raise
                confidence = round(prediction[1], 1)
                output = "REAL" if prediction[0] == 1 else "FAKE"
                print("Prediction:", prediction[0], "==", output, "Confidence:", confidence)
                print("<=== | Prediction Done | ===>")
                print("--- %s seconds ---" % (time.time() - start_time))

                # Uncomment if you want to create heat map images
                # for j in range(sequence_length):
                #     heatmap_images.append(plot_heat_map(j, model, video_dataset[i], './', video_file_name_only))

            # Render results
            context = {
                'preprocessed_images': preprocessed_images,
                'faces_cropped_images': faces_cropped_images,
                'heatmap_images': heatmap_images,
                'original_video': production_video_name,
                'models_location': os.path.join(settings.PROJECT_DIR, 'models'),
                'output': output,
                'confidence': confidence,
                'selected_model_name': selected_model_name,
                'sequence_length': sequence_length,
            }

            if settings.DEBUG:
                return render(request, predict_template_name, context)
            else:
                return render(request, predict_template_name, context)

        except Exception as e:
            error_message = str(e)
            print(f"Exception occurred during prediction: {error_message}")
            if "out of memory" in error_message.lower():
                return render(request, 'cuda_full.html')
            context = {
                'preprocessed_images': preprocessed_images,
                'faces_cropped_images': faces_cropped_images,
                'heatmap_images': [],
                'original_video': production_video_name,
                'prediction_error': error_message,
            }
            return render(request, predict_template_name, context)
def about(request):
    return render(request, about_template_name)

def handler404(request,exception):
    return render(request, '404.html', status=404)
def cuda_full(request):
    return render(request, 'cuda_full.html')
