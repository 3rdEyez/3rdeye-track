import cv2 as cv
import numpy as np
import time


class base_yunet_detector:
    #　onnx_path: onnx模型的路径
    #　input_size: 模型输入的尺寸
    def __init__(self, input_size, conf_thresh=0.6, iou_thresh=0.45):
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
    
    def __call__(self, rgb_img):
        
        self.img_scale, img = letterbox_topleft(rgb_img, self.input_size)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        cls_8, cls_16, cls_32, obj_8, obj_16, obj_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32 = \
            self._exec(img.copy())
        
        """
        np.save(f"cls_8.npy.{self.__class__.__name__}", cls_8)
        print("size of cls_8:", cls_8.shape)
        print("size of obj_8:", obj_8.shape)
        print("size of bbox_8:", bbox_8.shape)
        print("size of kps_8:", kps_8.shape)
        
        print("size of cls_16:", cls_16.shape)
        print("size of obj_16:", obj_16.shape)
        print("size of bbox_16:", bbox_16.shape)
        print("size of kps_16:", kps_16.shape)
        
        print("size of cls_32:", cls_32.shape)
        print("size of obj_32:", obj_32.shape)
        print("size of bbox_32:", bbox_32.shape)
        print("size of kps_32:", kps_32.shape)
        """
    
        prior3 = meshgrid(self.input_size[0] / 8, self.input_size[1] / 8, 8)
        prior4 = meshgrid(self.input_size[0] / 16, self.input_size[1] / 16, 16)
        prior5 = meshgrid(self.input_size[0] / 32, self.input_size[1] / 32, 32)
        
        bbox_8 = bbox_decode(bbox_8, prior3, 8)
        bbox_16 = bbox_decode(bbox_16, prior4, 16)
        bbox_32 = bbox_decode(bbox_32, prior5, 32)
        
        kps_8 = kps_decode(kps_8, prior3, 8)
        kps_16 = kps_decode(kps_16, prior4, 16)
        kps_32 = kps_decode(kps_32, prior5, 32)
        
        Cls = np.concatenate((cls_8, cls_16, cls_32), axis=1).flatten()
        Reg = np.concatenate((bbox_8, bbox_16, bbox_32), axis=1).reshape((-1, 4))
        Kps = np.concatenate((kps_8, kps_16, kps_32), axis=1).reshape((-1, 10))
        Obj = np.concatenate((obj_8, obj_16, obj_32), axis=1).flatten()
        faces = detection_output(Cls, Reg, Kps, Obj, self.conf_thresh, self.iou_thresh, 1000, 512)
        
        if len(faces) == 0:
            return None
        
        faces[:, :14] /= self.img_scale
        return faces

    def _exec(self, rgb_img):
        raise NotImplementedError


class openvino_yunet_detector(base_yunet_detector):
    #　onnx_path: onnx模型的路径
    #　input_size: 模型输入的尺寸
    def __init__(self, onnx_path, input_size, conf_thresh=0.6, iou_thresh=0.45):
        from openvino.runtime import Core
        super().__init__(input_size, conf_thresh, iou_thresh)
        self.core = Core()
        model = self.core.read_model(onnx_path)
        self.compiled_model = self.core.compile_model(model=model, device_name="CPU")

    def _exec(self, rgb_img):
        return list(self.compiled_model(rgb_img).values())


DATASET_PATH = '../../../media/COCO/coco_subset_20.txt'
class rknnsim_yunet_detector(base_yunet_detector):
    #　onnx_path: onnx模型的路径
    #　input_size: 模型输入的尺寸
    def __init__(self, onnx_path, target, input_size, conf_thresh=0.6, iou_thresh=0.45):
        super().__init__(input_size, conf_thresh, iou_thresh)
        from rknn.api import RKNN
        rknn = RKNN()
        
        # Pre-process config
        print('--> Config model')
        rknn.config(mean_values=[[0, 0, 0]], std_values=[
                        [1, 1, 1]], target_platform=target)
        print('done')
        
        # Load model
        ret = rknn.load_onnx(onnx_path)
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        print('done')

        # Build model
        print('--> Building model')
        ret = rknn.build(do_quantization=True, dataset=DATASET_PATH)
        if ret != 0:
            print('Build model failed!')
            exit(ret)
        print('done')
        
        # Init runtime
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        ret = rknn.init_runtime()
        self.rknn = rknn


    def _exec(self, rgb_img):
        # you have to use nhwc format
        rgb_img=rgb_img.transpose(0, 2, 3, 1)
        result = self.rknn.inference(inputs=[rgb_img])
        return result



class ncnn_yunet_detector(base_yunet_detector):
    import ncnn as pyncnn
    def __init__(self, model_param, model_bin, input_size, conf_thresh=0.6, iou_thresh=0.45):
        super().__init__(input_size, conf_thresh, iou_thresh)
        self.net = self.pyncnn.Net()
        self.net.load_param(model_param)
        self.net.load_model(model_bin)

    def _exec(self, rgb_img):
        rgb_img = rgb_img.astype(np.float32)
        mat_in = self.pyncnn.Mat(rgb_img)
        ex = self.net.create_extractor()
        ex.input(self.net.input_names()[0], mat_in)
        y = []
        for output_name in self.net.output_names():
            mat_out = self.pyncnn.Mat()
            ex.extract(output_name, mat_out)
            y.append(np.array(mat_out)[None])
        return y


def letterbox_topleft(img, size, color=(114, 114, 114)):
    shape = img.shape[:2]  # shape = [height, width]
    scale = min(size[0] / shape[1], size[1] / shape[0])
    nw = int(round(shape[1] * scale))
    nh = int(round(shape[0] * scale))
    
    img = cv.resize(img, (nw, nh), interpolation=cv.INTER_AREA)
    img_new = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    img_new[0:nh, 0:nw] = img
    return scale, img_new


def meshgrid(feature_width, feature_height, stride):
    # 生成水平和垂直坐标的范围
    x_range = (np.arange(feature_width) * stride)
    y_range = (np.arange(feature_height) * stride)

    # 生成坐标网格
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    # 将 x_grid 和 y_grid 堆叠成 (feature_height, feature_width, 2) 形状的数组
    out = np.stack((x_grid, y_grid), axis=-1)
    return out


# input dimension: (B, W*H, 4)
def bbox_decode(bbox, prior, stride):
    grid_w, grid_h = prior.shape[:2]
    # Reshape bbox to match the grid shape
    bbox = bbox.copy().reshape(-1, grid_w, grid_h, 4)
    
    # Calculate offsets using broadcasting
    offsets = np.zeros_like(bbox)
    offsets[..., :2] = prior[..., :2] + (bbox[..., :2] * stride)
    offsets[..., 2:] = np.exp(bbox[..., 2:]) * stride
    
    # Calculate the bounding box coordinates
    decoded_bbox = np.zeros_like(bbox)
    decoded_bbox[..., 0] = offsets[..., 0] - offsets[..., 2] / 2
    decoded_bbox[..., 1] = offsets[..., 1] - offsets[..., 3] / 2
    decoded_bbox[..., 2] = offsets[..., 0] + offsets[..., 2] / 2
    decoded_bbox[..., 3] = offsets[..., 1] + offsets[..., 3] / 2
    
    # Reshape back to the original shape
    return np.expand_dims(decoded_bbox.reshape(-1, 4), axis=0)


def kps_decode(kps, prior, stride):
    grid_w, grid_h = prior.shape[:2]
    # Reshape kps to match the grid shape
    kps = kps.copy().reshape(-1, grid_w, grid_h, 10)
    
    # Calculate offsets using broadcasting for all keypoints simultaneously
    cx = (kps[..., 0::2] * stride) + prior[:, :, 0, np.newaxis]
    cy = (kps[..., 1::2] * stride) + prior[:, :, 1, np.newaxis]
    
    # Update the keypoints coordinates
    kps[..., 0::2] = cx
    kps[..., 1::2] = cy
    
    # Reshape back to the original shape
    return np.expand_dims(kps.reshape(-1, 10), axis=0)

 
def xywh2xyxy(*box):
    """
    将xywh转换为左上角点和左下角点
    Args:
        box:
    Returns: x1y1x2y2
    """
    ret = [box[0] - box[2] // 2, box[1] - box[3] // 2, \
          box[0] + box[2] // 2, box[1] + box[3] // 2]
    return ret
 
def get_inter(box1, box2):
    """
    计算相交部分面积
    Args:
        box1: 第一个框
        box2: 第二个框
    Returns: 相交部分的面积
    """
    x1, y1, x2, y2 = xywh2xyxy(*box1)
    x3, y3, x4, y4 = xywh2xyxy(*box2)
    # 验证是否存在交集
    if x1 >= x4 or x2 <= x3:
        return 0
    if y1 >= y4 or y2 <= y3:
        return 0
    # 将x1,x2,x3,x4排序,因为已经验证了两个框相交,所以x3-x2就是交集的宽
    x_list = sorted([x1, x2, x3, x4])
    x_inter = x_list[2] - x_list[1]
    # 将y1,y2,y3,y4排序,因为已经验证了两个框相交,所以y3-y2就是交集的宽
    y_list = sorted([y1, y2, y3, y4])
    y_inter = y_list[2] - y_list[1]
    # 计算交集的面积
    inter = x_inter * y_inter
    return inter


def get_iou(box1, box2):
    """
    计算交并比： (A n B)/(A + B - A n B)
    Args:
        box1: 第一个框
        box2: 第二个框
    Returns:  # 返回交并比的值
    """
    box1_area = box1[2] * box1[3]  # 计算第一个框的面积
    box2_area = box2[2] * box2[3]  # 计算第二个框的面积
    inter_area = get_inter(box1, box2)
    union = box1_area + box2_area - inter_area   #(A n B)/(A + B - A n B)
    iou = inter_area / union
    return iou


def detection_output(Cls, Reg, Kps, Obj, conf_thresh, iou_thresh, topk=1000, keep_topk=512):
    conf = np.sqrt(Cls * Obj)
    keep_idx = np.where(conf > conf_thresh)
    Reg = Reg[keep_idx]
    Kps = Kps[keep_idx]
    conf = conf[keep_idx]
    score_bbox = []
    for i in range(len(keep_idx[0])):
        score_bbox.append([conf[i], Reg[i], Kps[i]])

    score_bbox = sorted(score_bbox, key=lambda x: -x[0])  # 按置信度从大到小排序
    if topk > - 1 and topk < len(score_bbox):
        score_bbox = score_bbox[:topk]
    
    # Do NMS
    keep_bbox = []
    while len(score_bbox) > 0:
        bb1 = score_bbox[0]
        keep = True
        for bb2 in keep_bbox:
            overlap = get_iou(bb1[1], bb2[1])
            if overlap > iou_thresh:
                keep = False
                break
        if keep:
            keep_bbox.append(bb1)
        score_bbox = score_bbox[1:]
    
    if len(keep_bbox) > keep_topk:
        keep_bbox = keep_bbox[:keep_topk]
    
    faces = []
    for item in keep_bbox:
        faces.append(np.concatenate((item[1], item[2], np.expand_dims(item[0], axis=0)), axis=-1))
    
    return np.array(faces)


def visualize(image, faces, print_flag=False, fps=None):
    output = image.copy()

    if fps:
        cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    if faces is not None:
        for idx, face in enumerate(faces):
            if print_flag:
                print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

            coords = face[:-1].astype(np.int32)
            # Draw face bounding box
            cv.rectangle(output, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
            # Draw landmarks
            cv.circle(output, (coords[4], coords[5]), 2, (255, 0, 0), 2)
            cv.circle(output, (coords[6], coords[7]), 2, (0, 0, 255), 2)
            cv.circle(output, (coords[8], coords[9]), 2, (0, 255, 0), 2)
            cv.circle(output, (coords[10], coords[11]), 2, (255, 0, 255), 2)
            cv.circle(output, (coords[12], coords[13]), 2, (0, 255, 255), 2)
            # Put score
            cv.putText(output, '{:.4f}'.format(face[-1]), (coords[0], coords[1]+15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    return output
