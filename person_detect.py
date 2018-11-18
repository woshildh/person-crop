from models import *
from utils.datasets import *
from utils.utils import *
import cv2

device="cpu"

class OPT(object):
	def __init__(self,input_path="data/samples/zidane.jpg",
		output_path="data/samples/zidane_output.jpg",img_size=416,
		cfg_path="cfg/yolov3.cfg",conf_thres=0.5,
		nms_thres=0.45,weights_path="weights/yolov3.pth"):

		self.weights_path=weights_path
		self.input_path=input_path
		self.output_path=output_path
		self.img_size=img_size
		self.cfg_path=cfg_path
		self.conf_thres=conf_thres
		self.nms_thres=nms_thres

opt=OPT()

def get_img(img_path,img_size=416):
	raw_img=cv2.imread(img_path)
	img,_,_,_=resize_square(raw_img,height=img_size,color=(127.5, 127.5, 127.5))
	img = img[:, :, ::-1].transpose(2, 0, 1) #转换BGR2RGB
	img = np.ascontiguousarray(img, dtype=np.float32)
	img=img/255
	return img,raw_img

def detect(opt):
	model = Darknet(opt.cfg_path, opt.img_size)
	print("get model ...")
	if opt.weights_path is not None:
		model.load_state_dict(torch.load(opt.weights_path,map_location="cpu")["model"])
		print("{} load succeed".format(opt.weights_path))

	model.to(device).eval()

	img,raw_img=get_img(opt.input_path,img_size=opt.img_size)
	with torch.no_grad():
		chip = torch.from_numpy(img).unsqueeze(0).to(device)
		pred = model(chip)
		pred = pred[pred[:, :, 4] > opt.conf_thres]

		if len(pred) > 0:
			detections = non_max_suppression(pred.unsqueeze(0), opt.conf_thres, opt.nms_thres)
		else:
			print("There is no object...")
			return -1
	
	detections=detections[0]
	# The amount of padding that was added
	pad_x = max(raw_img.shape[0] - raw_img.shape[1], 0) * (opt.img_size / max(raw_img.shape))
	pad_y = max(raw_img.shape[1] - raw_img.shape[0], 0) * (opt.img_size / max(raw_img.shape))
	# Image height and width after padding is removed
	unpad_h = opt.img_size - pad_y
	unpad_w = opt.img_size - pad_x
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
		plot_one_box([x1,y1,x2,y2],raw_img,label="Person:{:.3f}".format(conf),color=[0,0,255])
		
	cv2.imwrite(opt.output_path,raw_img)

if __name__=="__main__":
	detect(opt)


