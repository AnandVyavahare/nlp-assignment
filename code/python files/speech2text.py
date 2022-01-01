import os
import speech_recognition as sr
import ffmpeg
import subprocess
import boto3
import csv

session = boto3.Session(
aws_access_key_id= <ENTER AWS ACCESS KEY>,
aws_secret_access_key= <ENTER AWS SECRET KEY>
)


s3 = session.resource('s3')

my_bucket = s3.Bucket(<ENTER DEV, TRAIN, TEST BUCKET NAMES FOR EACH EXECUTION>)

mp4List = []
for s3_object in my_bucket.objects.all():
    filename = s3_object.key
    mp4List.append(filename)
	    
    my_bucket.download_file(s3_object.key, filename)


print("All Videos are downloaded")
outputList = []
for video in mp4List:
    dialog1 = []
    print(f"Video List {mp4List}")
    mp3Name = video.split('.')[0]
    command2mp3 = f"ffmpeg -i {video} {mp3Name}.mp3"
    command2wav = f"ffmpeg -i {mp3Name}.mp3 {mp3Name}.wav"
    os.system(command2mp3)
    os.system(command2wav)
    r = sr.Recognizer()
    print(f"Audio File {mp3Name}.wav")
    with sr.AudioFile(f"{mp3Name}.wav") as source:
        r.adjust_for_ambient_noise(source,duration=0.5)
        audio = r.listen(source)
        print("^" * 30)
    try:
        query = r.recognize_google(audio, language='en-US',show_all = False)
        print("#" * 30)
        print(query)
        dialog1.append(query)
        outputList.append(dialog1)
        print("@" * 30)
    except Exception as e:
        print("$" * 30)
        print(e)

with open("output.csv", 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(outputList)
print(outputList)
