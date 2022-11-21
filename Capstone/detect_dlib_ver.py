import cv2
import dlib
import numpy as np
from model import Net
import torch
from imutils import face_utils
import winsound as sd

IMG_SIZE = (34,26)
PATH = './weights/trained.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#얼굴인식
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

model = Net()
model.load_state_dict(torch.load(PATH))
model.eval() #Dropout, Batchnorm등의 기능을 비활성화 하여 추론할 때의 모드로 작동하도록 조정

n_count = 0 #프레임 저장 변수


def beepsound(): #비프음 함수
  fr = 2000
  du = 100
  sd.Beep(fr,du)

def crop_eye(img, eye_points):  #눈동자 추출 함수
  x1, y1 = np.amin(eye_points, axis=0)
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

  eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect


def predict(pred):  # 모델 예측 함수
  pred = pred.transpose(1, 3).transpose(2, 3)

  outputs = model(pred)

  pred_tag = torch.round(torch.sigmoid(outputs))

  return pred_tag

cap = cv2.VideoCapture(0) #캠 넘버

while cap.isOpened():
  ret, img_ori = cap.read()

  if not ret:
    break

# 사이즈 조절 및 rgb image를 gray scale로 전환
  img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)

  img = img_ori.copy()
  img=cv2.flip(img,1) #거울처럼 좌우반전
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  faces = detector(gray)

  for face in faces:
    shapes = predictor(gray, face)
    shapes = face_utils.shape_to_np(shapes)

    eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
    eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])


    eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
    eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
    eye_img_r = cv2.flip(eye_img_r, flipCode=1)
#눈만 인식한 결과 보기
    #cv2.imshow('l', eye_img_l)
    #cv2.imshow('r', eye_img_r)
    eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)
    eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32)

#numpy를 torch로 변환
    eye_input_l = torch.from_numpy(eye_input_l)
    eye_input_r = torch.from_numpy(eye_input_r)

    pred_l = predict(eye_input_l)
    pred_r = predict(eye_input_r)

    if pred_l.item() == 0.0 and pred_r.item() == 0.0:
      n_count+=1

    else:
      n_count = 0

#50프레임 동안 눈을 감으면 Wake up 표시와 함꼐 비프음 1초 무한재생
    if n_count > 50:
      cv2.putText(img,"Wake up", (120,160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2) #(파랑,녹색,빨강)
      beepsound()

    # visualize
    state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'
    state_r = 'O %.1f' if pred_r > 0.1 else '- %.1f'

    state_l = state_l % pred_l
    state_r = state_r % pred_r

    cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255,255,255), thickness=2)
    cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255,255,255), thickness=2)

    cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
  #좌우 눈을 0과 1로 판별
    # print(state_l)
    # print(state_r)

  cv2.namedWindow('result',cv2.WINDOW_NORMAL)
  #cv2.resizeWindow('result',1280,720)
  cv2.imshow('result', img)

  #cv2.waitKey(0)

  if cv2.waitKey(1) == ord('q'):
    break
