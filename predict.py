import json
import os


def predict_one(yolo_model, image_path: str) -> list:

    """
    Функция для обнаружения баундин боксов на одном изображении.
    Args:
        yolo_model - модель Yolov5;
        image_path - путь к изображению.
    Returns:
        bounding_boxes_list - список с координатами баундин боксов в формате COCO_json bbox. 
    """

    bounding_boxes_list = []

    prediction = yolo_model(image_path)
    predict_data = prediction.pandas().xyxy[0].to_numpy()[:, :-3].tolist()
    for bbox in predict_data:
        pred_bbox = {
            'area': 31006.108000000015,
            'attributes': {'occluded': False},
            'bbox': [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]],
            'category_id': 1,
            'id': 1,
            'image_id': 1,
            'iscrowd': 0,
            'segmentation': []
            }
        bounding_boxes_list.append(pred_bbox)
    
    return bounding_boxes_list


def predict_submission(yolo_model, example_path, images_paths) -> list:

    """
    Функция для обнаружения баундин боксов на одном изображении.
    Args:
        yolo_model - модель Yolov5;
        example_path - путь к submission_example.json;
        images_paths - путь к папке с тестовыми изображениями.
    Returns:
        bounding_boxes_list - список с координатами баундин боксов в формате COCO_json bbox. 
    """

    with open(example_path, 'r') as sub:
        submission = json.load(sub)
    bbox_id = 1
    bounding_boxes_list = []
    for i in range(len(submission['images'])):
        predictions = yolo_model(os.path.join(images_paths, submission['images'][i]['file_name']))
        predict_data = predictions.pandas().xyxy[0].to_numpy()[:, :-3].tolist()

        for bbox in predict_data:
            pred_bbox = {
                'area': 31006.108000000015,
                'attributes': {'occluded': False},
                'bbox': [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]],
                'category_id': 1,
                'id': bbox_id,
                'image_id': submission['images'][i]['id'],
                'iscrowd': 0,
                'segmentation': []
                }
            bounding_boxes_list.append(pred_bbox)
            bbox_id += 1
    submission['annotations'] = bounding_boxes_list

    return submission
