#import required libraries
import cv2
import boto3
import os
import io
from PIL import Image, ImageDraw, ExifTags, ImageColor, ImageFont
import matplotlib.pyplot as plt


# Read the video from specified path
cam = cv2.VideoCapture("/Users/admin/Downloads/VID_20200815_165535.mp4")

try:

    # creating a folder named data
    if not os.path.exists('data'):
        os.makedirs('data')

# if not created then raise error
except OSError:
    print ('Error: Creating directory of data')

# start extractin frames from videos
currentframe = 0

while(True):

    # reading from frame
    ret,frame = cam.read()

    if ret:
        # if video is still left continue creating images
        #every frame is resized
        #it's important that all images are of same size, as at end when we have to create a video it should be of same size otherwise you will get an error
        width= 416
        height= 416
        dsize = (width, height)
        output = cv2.resize(frame, dsize)



        #save each frame as 'frame0' and so on in folder named data
        name = '/Users/admin/Desktop/results-for-video/data/frame' + str(currentframe) + '.jpg'
        print ('Creating...' + name)

        # writing the extracted images to local folder data
        cv2.imwrite(name, output)


        #now first frame as 'frame0' is going to upload in s3
        def upload_files(path):
            session = boto3.Session(
                #replace access key, secret key and region with your specific ones
                aws_access_key_id='YOUR ACCESS KEY',
                aws_secret_access_key='YOUR SECRET KEY',
                region_name='YOUR REGION'
        )
            s3 = session.resource('s3')

            #replace argument with bucket in which you want to save the frames
            bucket = s3.Bucket('images-for-detection')

            for subdir, dirs, files in os.walk(path):
                for file in files:
                    full_path = os.path.join(subdir, file)
                    with open(full_path, 'rb') as data:
                        bucket.put_object(Key=full_path[len(path)+1:], Body=data)


        #calling function for inference
        def show_custom_labels(model,bucket,photo, min_confidence):


            #calling rekognition model
            client=boto3.client('rekognition')

            #Load image from S3 bucket
            s3_connection = boto3.resource('s3')

            s3_object = s3_connection.Object(bucket,photo)
            s3_response = s3_object.get()

            stream = io.BytesIO(s3_response['Body'].read())
            image=Image.open(stream)

            #Call DetectCustomLabels
            response = client.detect_custom_labels(Image={'S3Object': {'Bucket': bucket, 'Name': photo}},MinConfidence=min_confidence,ProjectVersionArn=model)

            #show image of infered frame0
            imgWidth, imgHeight = image.size
            print(imgWidth)
            print(imgHeight)
            fig,ax = plt.subplots(1, 1, figsize=[8,8])

            draw = ImageDraw.Draw(image)

            #calculate and display bounding boxes for each detected custom label
            print('Detected custom labels for ' + photo)
            for customLabel in response['CustomLabels']:

                #print name of label and its confidence
                print('Label ' + str(customLabel['Name']))
                print('Confidence ' + str(customLabel['Confidence']))
                if 'Geometry' in customLabel:

                    #defines geometry of box
                    box = customLabel['Geometry']['BoundingBox']
                    left = imgWidth * box['Left']
                    top = imgHeight * box['Top']
                    width = imgWidth * box['Width']
                    height = imgHeight * box['Height']

                    #write label name at top left of box
                    draw.text((left,top), customLabel['Name'])

                    #print geometry paramters as json format
                    print('Left: ' + '{0:.0f}'.format(left))
                    print('Top: ' + '{0:.0f}'.format(top))
                    print('Label Width: ' + "{0:.0f}".format(width))
                    print('Label Height: ' + "{0:.0f}".format(height))

                    points = ((left,top),(left + width, top),(left + width, top + height),(left , top + height),(left, top))

                    #if label is a paper, draws a box with width of 2 and color corresponding to color code '#00d400'
                    #if need thicker box, increase width, all labels have different color of boxes according to their color codes
                    if str(customLabel['Name']) == 'paper':
                        draw.line(points, fill='#00d400', width=2)
                    if str(customLabel['Name']) == 'cardboard':
                        draw.line(points, fill='#bc4100', width=2)
                    if str(customLabel['Name']) == 'tissue':
                        draw.line(points, fill='#fffe1c', width=2)
                    if str(customLabel['Name']) == 'can':
                        draw.line(points, fill='#221600', width=2)
                    if str(customLabel['Name']) == 'bottle':
                        draw.line(points, fill='#4b5a00', width=2)
                    if str(customLabel['Name']) == 'plastic_bag':
                        draw.line(points, fill='#ff32a7', width=2)
                    if str(customLabel['Name']) == 'wrapper':
                        draw.line(points, fill='#0004bf', width=2)
                    if str(customLabel['Name']) == 'trash':
                        draw.line(points, fill='#ffddfd', width=2)
                    if str(customLabel['Name']) == 'wood':
                        draw.line(points, fill='#030030', width=2)
                    if str(customLabel['Name']) == 'compost':
                        draw.line(points, fill='#cd4f39', width=2)

                    #set axis to off and margins to zero
                    ax.imshow(image)
                    plt.axis("off")

                    plt.margins(0,0)

                    #save each infered frame as image0 and so on locally in folder where this python file is
                    #for making video, these infered images should be in alphabatically arranged
                    #so i put if statement, as my frames were about 130 for video, so zeros can be added in start to keep them alphabatically

                    my_dpi=138

                    if currentframe in range(0,10):
                        plt.savefig('00'+str(currentframe)+'image'+'.jpg', transparent = True, bbox_inches='tight', pad_inches=0,dpi=my_dpi)
                    if currentframe in range(10,100):
                        plt.savefig('0'+str(currentframe)+'image'+'.jpg', transparent = True, bbox_inches='tight', pad_inches=0,dpi=my_dpi)
                    if currentframe in range(100,1000):
                        plt.savefig(str(currentframe)+'image'+'.jpg', transparent = True, bbox_inches='tight', pad_inches=0,dpi=my_dpi)
                    #if your images are more than 1000 add one more if statement and one more zero in each if statement

            return len(response['CustomLabels'])

        def main():

            #adding frames from folder data to s3
            upload_files('/Users/admin/Desktop/results-for-video/data')

            #bucket that has all frames
            bucket="images-for-detection"

            #photo="frame0.jpg"
            photo= 'frame'+str(currentframe)+'.jpg'


            #model arn with 10 classes
            model='YOUR MODEL ARN'

            min_confidence=95

            label_count=show_custom_labels(model,bucket,photo, min_confidence)
            print("Custom labels detected: " + str(label_count))



        if __name__ == "__main__":
            main()

        # increasing counter so that it will show how many frames are created
        #counter will run untill all frames are done with inference and saved as image'i'
        currentframe += 1
    else:
        break

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()
