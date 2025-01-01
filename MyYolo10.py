import cv2, os, random
import supervision as sv
import numpy as np
from moviepy.editor import ImageSequenceClip
# from mystatistics import makeStatistics

  
class MyYolo:
    def __init__(self, model, taskDir, fileName, log, hash):
        splittedName = fileName.split('.')
        self.model = model
        self.log = log
        self.hash = hash
        self.videoName=fileName.split('_')[0]
        self.basePath=taskDir
        self.fileName = fileName
        self.format = splittedName[-1]
        self.nameNdate = splittedName[0]
        self.inputDir = taskDir + f'/input/{fileName}'
        self.outDir = taskDir  + '/output'
        self.outFileDir = f"{self.outDir}/tagged_{fileName}"
        self.inputFrames = taskDir + '/frames/frames'
        self.taggedFrame = taskDir + '/frames/taggedframes'
        self.frameCount = -1
        self.prompt={'current': ''}
        self.promptSet=set()
        self.nextStep= [-1]
        self.processing=''
        self.statistics=''

    def getResDetails(self):
        return {
            'name': self.videoName ,
            'objects': list(self.promptSet),
            'prompts': self.prompt,
            'inDir' : self.inputDir,
            'outDir' : self.outFileDir,
            'stats': self.statistics,
            'fps' : self.fps,
            'nameNdate' : self.nameNdate
        }

    def logSet(self, log):
        self.log = log

    def start(self):
        try:   
            self.setStage(1)
        except Exception as e:
            raise Exception(f"split video to frame: {e}")
        try:  
            self.setStage(2) 
        except Exception as e:
            raise Exception(f"yolo prediction and detection: {e}")
        try:  
            self.setStage(3)
        except Exception as e:
            raise Exception(f"combine frames to video: {e}")
        try:
            self.setStage(4)
        except Exception as e:
            raise Exception(f"statistics analyzing: {e}")
        

    def split_video_to_frames(self):
        video_path = self.inputDir
        output_folder = self.inputFrames
    # Open the video file
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        count = 0
        self.fps = vidcap.get(cv2.CAP_PROP_FPS)

        
        if not success:
            raise Exception("Error in parsing input video to frames")
 

        # Iterate through the video frames
        while success:
            # Save frame as JPEG file
            cv2.imwrite(f"{output_folder}/frame_{count}.jpg", image)
            success, image = vidcap.read()
            count += 1
            
        return count
    
    def framesToVideo(self,fps=24):
        frame_folder = self.taggedFrame
        output_video_path = self.outFileDir
        fps = self.fps
        # Get all .jpg frame files sorted by number in filename
        frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith('.jpg')], key=lambda x: int(x.split('_')[1].split('.')[0]))

        # Create full paths to each frame
        frame_paths = [os.path.join(frame_folder, frame_file) for frame_file in frame_files]

        # Create the video clip from the image sequence
        clip = ImageSequenceClip(frame_paths, fps=fps)

        # Write the video to the output path in mp4 format (MoviePy uses libx264 by default)
        clip.write_videofile(output_video_path, codec='libx264')
        
        # Optional logging (if self.log exists)
        # self.log.info('Video written to output folder successfully')

    def predictAndDetect(self):
        prompt=[]
        frame_files = sorted([f for f in os.listdir(self.inputFrames) if f.endswith('.jpg')], key=lambda x: int(x.split('_')[1].split('.')[0]))
        k, m = divmod(len(frame_files), 10)
        for i in range(10):
            self.nextStep[0]=i+1
            tenth_frame_files = frame_files[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
            for f in tenth_frame_files:
                frame = cv2.imread(f"{self.inputFrames}/{f}")
                results = self.model(frame, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)
                annotated_image = sv.BoxAnnotator().annotate(
                    scene=frame, detections=detections)

                annotated_image = sv.LabelAnnotator().annotate(
                    scene=annotated_image, detections=detections)
                unique_elements, counts = np.unique(detections.data['class_name'].tolist(), return_counts=True)
                unique_elements = unique_elements.tolist()
                counts = counts.tolist() 
                self.promptSet.update(unique_elements)
                frameObjects = dict(zip(unique_elements, counts))
                self.prompt[f] = frameObjects
                if unique_elements:
                    self.prompt['current'] = random.choice(unique_elements)

                cv2.imwrite(f"{self.taggedFrame}/{f}", annotated_image)
            self.log.info(f'Detection of {i*10}% completed')
        self.prompt.pop('current', None)
        self.nextStep[0]=-1
        
    def setStage(self, stage):
        if stage == 1:
            self.log.info('Start splitting into frames')
            self.updateProcessing("Splitting video into frames")
            self.frameCount = self.split_video_to_frames()
        elif stage == 2:
            self.log.info('Frames object detection started')
            self.updateProcessing("Looking for objects")
            self.predictAndDetect()
        elif stage == 3:
            self.updateProcessing("Finishing up")
            self.log.info('Packing frames into tagged video')
            self.framesToVideo()
        elif stage == 4:
            self.log.info('Detected video is ready')
            self.statistics = self.makeStatistics()
        elif stage == 5:
            self.log.info('Data inserted to DB')

    def updateProcessing(self, update):
        self.processing = update

    def generateDbData(self):
        #return {"name":self.nameNdate ,"data":self.prompt,"promptDistinct": list(self.promptSet), "videoFormat": self.format, "frameCount": self.frameCount,"fps":self.fps, "hash256": self.hash, "statistics":self.statistics}
        return {"name":self.nameNdate ,"promptDistinct": list(self.promptSet), "videoFormat": self.format, "frameCount": self.frameCount,"fps":self.fps, "hash256": self.hash, "statistics":self.statistics}

    def updateDB(self,func):
        if func(self.generateDbData()):
            self.setStage(5)
            return True
        return False
    
    def makeStatistics(self):
        frame_details = self.prompt
        arr = self.promptSet
        numOfFrames = len(frame_details)
        object_statistics = {key: {"frameCount" : 0 , "framesShown" : [], "minInFrame": {"min":None ,"frame":None}, "maxInFrame": {"max":None ,"frame":None}, "maxSequence":{"frames":[], "count" : -1}} for key in arr}
        objectCounts = {}
        countObjectsTypesInFrame={}
        frame_statistics = {"rawData":frame_details}
        video_statistics = {"objectCounts":{}}

        for index, (frame,objects) in enumerate(frame_details.items()):
            for obj,count in objects.items():
                if obj in objectCounts: # counter number of apperance in video
                    objectCounts[obj] += count
                else:
                    objectCounts[obj] = count
                # Update minInFrame
                if not object_statistics[obj]["minInFrame"]["min"] or count < object_statistics[obj]["minInFrame"]["min"]:
                    object_statistics[obj]["minInFrame"]["min"] = count
                    object_statistics[obj]["minInFrame"]["frame"] = index

            
                # Update maxInFrame
                if not object_statistics[obj]["maxInFrame"]["max"]  or count > object_statistics[obj]["maxInFrame"]["max"]:
                    object_statistics[obj]["maxInFrame"]["max"] = count
                    object_statistics[obj]["maxInFrame"]["frame"] = index
        video_statistics["objectCounts"]=objectCounts
        
        for index, (frame, details) in enumerate(frame_details.items()):
            for vehicle_type in arr:
                if vehicle_type in list(details):
                    object_statistics[vehicle_type]["frameCount"] += 1
                    #extract frame number
                    object_statistics[vehicle_type]["framesShown"].append(index)

        #count distinct objects by frame
        for key in frame_details:
            countObjectsTypesInFrame[key]=len(frame_details[key])


    #check sequence of consecutive objects in frames 
        for key in object_statistics: # in arr
            seq=self.__largest_consecutive_sequence(object_statistics[key]["framesShown"])
            object_statistics[key]["maxSequence"]["frames"]= seq
            object_statistics[key]["maxSequence"]["count"]= len(seq)
        

        # הדפסת הסטטיסטיקות
        print("object_statistics:")
        for vehicle_type, properties in object_statistics.items():
                print(f"Total data about {vehicle_type}:")
                for property, value in properties.items():
                    print(f"Total {property}: {value}")
        print(object_statistics)
        return {"frame":frame_statistics,"object":object_statistics,"video":video_statistics}

    def __largest_consecutive_sequence(self,sorted_list):
        if not sorted_list:
            return []

        # Initialize variables to keep track of the longest sequence
        longest_sequence = []
        current_sequence = [sorted_list[0]]

        # Iterate through the sorted list
        for i in range(1, len(sorted_list)):
            if sorted_list[i] == sorted_list[i-1] + 1:
                # If the current number is consecutive, add it to the current sequence
                current_sequence.append(sorted_list[i])
            else:
                # If the current sequence ends, check if it's the longest
                if len(current_sequence) > len(longest_sequence):
                    longest_sequence = current_sequence
                # Reset the current sequence
                current_sequence = [sorted_list[i]]

        # Check the last sequence
        if len(current_sequence) > len(longest_sequence):
            longest_sequence = current_sequence

        return longest_sequence


        
            