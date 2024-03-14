from ultralytics import YOLO

if __name__ == '__main__':
    import matplotlib

    matplotlib.use('TkAgg')
    # Load a model
    model = YOLO('./yolov8n-seg.yaml').load(
   r'E:\workspace\pycharm\yolo\ultralytics-main\runs\segment\train39\weights\best.pt')
    # model = YOLO('./yolov8n-seg.yaml')
    # build a new model from YAML
    # model = YOLO('./yolov8n-seg.yaml') .load(r'E:\workspace\pycharm\yolo\ultralytics-main\runs\segment-back\train45\weights\best.pt') # build a new model from YAML
    # model = YOLO('./yolov8n-seg.pt')  # load a pretrained model (recommended for training)
    # model = YOLO(r'E:\workspace\pycharm\yolo\ultralytics-main\runs\segment\train15\wei
    # ghts\best.pt')  # load a pretrained model (recommended for training)
    # model = YOLO('./yolov8n-seg.yaml').load('./           yolov8s.pt')  # build from YAML and transfer weights
    # model = YOLO('./yolov8n-seg.yaml').load('./yolov8s.pt')  # build from YAML and transfer weights

    # Train the model
    results = model.train(data='./coco128-seg-seal.yaml', epochs=200, imgsz=640, resume=True)
