# Inference-of-Rekognition-model-by-accessing-laptop-camera-(images-and-videos)-
My post is about making a garbage sorter, an object detection approach using AWS Rekognition. By training the model for 20 times step by step, I got a F1 score of 0.943 with 620 images.
My images are resized to 416 by 416 which is AWS rekognition requirement. Images were captured by phone and were augmented too. 
In this post i will show you how can you call rekognition model for inference and can show inference by accessing your laptop camera (taking image and videos).
This live demo shows how can one take image with laptop camera, store it in S3 bucket, access Rekognition model, do inference and get results with bounding boxes and labels.
One can also show results by video. I will explain how to convert video to frames, do inference on every frame and at the end show output as a video.
Follow my medium post for step by step project implementation.
https://medium.com/@durafshanafshi/building-a-smart-garbage-sorter-using-rekognition-along-with-inference-76fc33082165
