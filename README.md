# person-crop
这是一个基于yolov3的行人检测与切割demo

算法代码主要来自于: https://github.com/ultralytics/yolov3

## 使用方法:
(1).下载 https://storage.googleapis.com/ultralytics/yolov3.pt, 并命名为yolov3.pth放入weights目录下，也可以放到自己的目录下。  
(2).定义person_crop中的PersonCrop的对象。  
(3).使用PersonCrop中的crop_person_by_img_path(self,img_path)或者通过crop_person_by_img_mat(self,img)分别传入图片路径和opencv的图像数组。  
(4).返回结果包括person_pos_list和person_img_list，之后可以进行自己的处理。  

实例见person_crop.py中的例子
