
import cv2
import os
import joblib
import pickle
import numpy as np
import streamlit as st 
from PIL import Image, ImageDraw, ImageFont
import torch
from facenet_pytorch import MTCNN, fixed_image_standardization, InceptionResnetV1
from torchvision import datasets, transforms


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Store the initial value of widgets in session state
if "model" not in st.session_state:
    st.session_state.model = "svm.sav"

with st.spinner("Loading Model..."):
    clf = joblib.load('./models/' + st.session_state.model)
    
IDX_TO_CLASS = ['liam', 'michael', 'ruben', 'wout']

mtcnn = MTCNN(keep_all=True, min_face_size=70, device=device)
    
standard_transform = transforms.Compose([
                                transforms.Resize((160, 160)),
                                np.float32, 
                                transforms.ToTensor(),
                                fixed_image_standardization
])
    
model = InceptionResnetV1(pretrained='vggface2', dropout_prob=0.6, device=device).eval()

def get_video_embedding(model, x): 
    embeds = model(x.to(device))
    return embeds.detach().cpu().numpy()
    
def face_extract(model, clf, frame, boxes):
    names, prob = [], []
    if len(boxes):
        x = torch.stack([standard_transform(frame.crop(b)) for b in boxes])
        embeds = get_video_embedding(model, x)
        idx, prob = clf.predict(embeds), clf.predict_proba(embeds).max(axis=1)
        
        names = [IDX_TO_CLASS[idx_] for idx_ in idx]
    return names, prob 

def diag(x1, y1, x2, y2):
    return np.linalg.norm([x2 - x1, y2 - y1])


def square(x1, y1, x2, y2):
    return abs(x2 - x1) * abs(y2 - y1)

def isOverlap(rect1, rect2):
    x1, x2 = rect1[0], rect1[2]
    y1, y2 = rect1[1], rect1[3]

    x1_, x2_ = rect2[0], rect2[2]
    y1_, y2_ = rect2[1], rect2[3]

    if x1 > x2_ or x2 < x1_: return False 
    if y1 > y2_ or y2 < y1_: return False
  
    rght, lft = x1 < x1_ < x2, x1_ < x1 < x2_
    d1, d2 = 0, diag(x1_, y1_, x2_, y2_)
    threshold = 0.5

    if rght and y1 < y1_: d1 = diag(x1_, y1_, x2, y2)
    elif rght and y1 > y1_: d1 = diag(x1_, y2_, x2, y1)
    elif lft and y1 < y1_: d1 = diag(x2_, y1_, x1, y2) 
    elif lft and y1 > y1_: d1 = diag(x2_, y2_, x1, y1)

    if d1 / d2 >= threshold and square(x1, y1, x2, y2) < square(x1_, y1_, x2_, y2_): return True
    return False

def draw_box(draw, boxes, names, probs, min_p=0.89):
    font = ImageFont.truetype(os.path.join('arial.ttf'), size=22)

    not_overlap_inds = []
    for i in range(len(boxes)): 
        not_overlap = True
        for box2 in boxes: 
            if np.all(boxes[i] == box2): continue 
            not_overlap = not isOverlap(boxes[i], box2)   
            if not not_overlap: break 
        if not_overlap: not_overlap_inds.append(i)

    boxes = [boxes[i] for i in not_overlap_inds] 
    probs = [probs[i] for i in not_overlap_inds]
    for box, name, prob in zip(boxes, names, probs):
        if prob >= min_p: 
            draw.rectangle(box.tolist(), outline=(255, 255, 255), width=5)
            x1, y1, _, _ = box
            text_width, text_height = font.getsize(f'{name}')
            draw.rectangle(((x1, y1 - text_height), (x1 + text_width, y1)), fill='white')
            draw.text((x1, y1 - text_height), f'{name}: {prob:0.5f}', (24, 12, 30), font) 
            
    return boxes, probs 
    
def preprocess_image(detector, face_extractor, clf, path, transform=None):
    if not transform: transform = lambda x: x.resize((1280, 1280)) if (np.array(x.size) > 2000).all() else x

    capture = path.convert('RGB')
    

    iframe = transform(capture)
   
    boxes, probs = detector.detect(iframe)
    if boxes is None: boxes, probs = [], []
    names, prob = face_extract(face_extractor, clf, iframe, boxes)
    
    frame_draw = iframe.copy()
    draw = ImageDraw.Draw(frame_draw)

    boxes, probs = draw_box(draw, boxes, names, probs)
        
    return frame_draw.resize((620, 480), Image.BILINEAR), names, prob


def detect(clf, img):
    frame, names, prob = preprocess_image(mtcnn, model, clf, img)
    # while prob < 0.89:
    #     print(prob)
    #     frame, names, prob = preprocess_image(mtcnn, model, clf, img)
    
    st.image(frame, channels="RGB")
    st.success(f"Found {names} | {prob}")
    for name, prob in zip(names, prob):
        st.write(f"{name} | {prob}")
        

# camera = cv2.VideoCapture(0)

# FRAME_WINDOW = st.image([])

# run = st.checkbox('Run')

# while run:
#     _, frame = camera.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     FRAME_WINDOW.image(frame)
# else:
#     st.write('Stopped')


img_file_buffer = st.camera_input("Start Chatting", key="camera")  
st.write(st.session_state.model)
    
if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    
    img = Image.open(img_file_buffer)
    st.info("model loaded")
    detect(clf, img)
   
    

    