from models import *
from utils.datasets import *
from utils.utils import *
import cv2,os

class PersonCrop(object):
	def __init__(self,img_size=416,cfg_path="cfg/yolov3.cfg",class_path="data/coco.names",
		conf_thres=0.5,nms_thres=0.45,weights_path="weights/yolov3.pth",device="cpu"):
		
		self.device=device
		self.weights_path=weights_path
		self.img_size=img_size
		self.cfg_path=cfg_path
		self.class_path=class_path
		self.conf_thres=conf_thres
		self.nms_thres=nms_thres
		self.load_model()

	def load_model(self):
		self.model = Darknet(self.cfg_path, self.img_size)
		print("get model ...")
		if os.path.exists(self.weights_path):
			self.model.load_state_dict(torch.load(self.weights_path,map_location="cpu")["model"])
			print("{} load succeed".format(self.weights_path))

		self.model.to(self.device).eval()
	
	def get_img(self,raw_img):
		img,_,_,_=resize_square(raw_img,height=self.img_size,color=(127.5, 127.5, 127.5))
		img = img[:, :, ::-1].transpose(2, 0, 1) #转换BGR2RGB
		img = np.ascontiguousarray(img, dtype=np.float32)
		img=img/255
		return img,raw_img
	

	def crop_person_by_img_path(self,img_path):
		if os.path.exists(img_path):
			img=cv2.imread(img_path)
		else:
			print("{} doesn't exists".format(img_path))
			return
		img,raw_img=self.get_img(img)
		person_pos_list=self.get_person_list(img,raw_img)
		person_img_list=[]
		if len(person_pos_list)==0:
			return [],[]
		for i in range(len(person_pos_list)):
			x1,y1,x2,y2=person_pos_list[i]
			x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
			x1,y1,x2,y2=max([x1,0]),max([y1,0]),min(x2,raw_img.shape[1]),min(y2,raw_img.shape[0])
			person=raw_img[y1:y2,x1:x2,:]
			person_img_list.append(person)
		return person_pos_list,person_img_list

	def crop_person_by_img_mat(self,img):
		img,raw_img=self.get_img(img)
		person_pos_list=self.get_person_list(img,raw_img)
		person_img_list=[]
		if len(person_pos_list)==0:
			return [],[]
		for i in range(len(person_pos_list)):
			x1,y1,x2,y2=person_pos_list[i]
			x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
			x1,y1,x2,y2=max([x1,0]),max([y1,0]),min(x2,raw_img.shape[1]),min(y2,raw_img.shape[0])
			person=raw_img[y1:y2,x1:x2,:]
			person_img_list.append(person)
		return person_pos_list,person_img_list

	def get_person_list(self,img,raw_img):
		with torch.no_grad():
			chip = torch.from_numpy(img).unsqueeze(0).to(self.device)
			pred = self.model(chip)
			pred = pred[pred[:, :, 4] > self.conf_thres]

			if len(pred) > 0:
				detections = non_max_suppression(pred.unsqueeze(0), self.conf_thres, self.nms_thres)
			else:
				print("There is no object...")
				return []
		
		detections=detections[0]
		# The amount of padding that was added
		pad_x = max(raw_img.shape[0] - raw_img.shape[1], 0) * (self.img_size / max(raw_img.shape))
		pad_y = max(raw_img.shape[1] - raw_img.shape[0], 0) * (self.img_size / max(raw_img.shape))
		# Image height and width after padding is removed
		unpad_h = self.img_size - pad_y
		unpad_w = self.img_size - pad_x
		person_list=[]
		for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
			#根据人进行筛选
			if cls_pred!=0:
				continue
			# Rescale coordinates to original dimensions
			box_h = ((y2 - y1) / unpad_h) * raw_img.shape[0]
			box_w = ((x2 - x1) / unpad_w) * raw_img.shape[1]
			y1 = (((y1 - pad_y // 2) / unpad_h) * raw_img.shape[0]).round().item()
			x1 = (((x1 - pad_x // 2) / unpad_w) * raw_img.shape[1]).round().item()
			x2 = (x1 + box_w).round().item()
			y2 = (y1 + box_h).round().item()
			x1, y1, x2, y2 = max(x1, 0), max(y1, 0), max(x2, 0), max(y2, 0)
			person_list.append([x1,y1,x2,y2])
		return person_list	

if __name__=="__main__":
	crop=PersonCrop()
	image=cv2.imread("./samples/multi_person.jpg")
	person_pos_list,person_img_list=crop.crop_person_by_img_mat(image)
	print(person_pos_list)
	print(len(person_img_list))
	for i,img in enumerate(person_img_list):
		cv2.imwrite("./{}_mat.jpg".format(i),img)