import torch
import cv2
from utils.feature_extractor import featureExtractor
from utils.data_loader import TestDataset
from torch.utils.data import DataLoader
import numpy as np
from time import time
import onnxruntime as ort

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def predict(session, img): 
    #Run inference using ONNX model
    #Parameters:
    #    session (onnxruntime.InferenceSession): inference session based on provided onnx model 
    #    img (str): path to image 

    #Returns:
    #    prediction (bool): True if image blurred, False otherwise
    #    score (float): prediction score returned by model 

    feature_extractor = featureExtractor()
    accumulator = []
    threshold=0.5
    feature_extractor.resize_image(img, np.shape(img)[0], np.shape(img)[1])

    # compute the image ROI using local entropy filter
    feature_extractor.compute_roi()

    # # extract the blur features using DCT transform coefficients
    extracted_features = feature_extractor.extract_feature()
    extracted_features = np.array(extracted_features)

    if(len(extracted_features) == 0):
        print("Error processing features")

    test_data_loader = DataLoader(TestDataset(extracted_features), batch_size=1, shuffle=False)

    for batch_num, input_data in enumerate(test_data_loader):
        x = input_data
        x = x.float()
        x = np.resize(x,(1,x.shape[1]))
        output = session.run(['output'], {'input': x})
        output = torch.tensor(output[0])
        _, predicted_label = torch.max(output, 1)
        accumulator.append(predicted_label.item())
    
    score = np.mean(accumulator)
    prediction_label = score < threshold
    return prediction_label, score


if __name__ == '__main__':

    img = cv2.imread("dataset/test/0_1.jpg", 0)
    ort_sess = ort.InferenceSession('blur.onnx', None)
    pred, score = predict(ort_sess, img)
    print(pred, score)
    
