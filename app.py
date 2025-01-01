import random, hashlib, os, torch, base64
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from datetime import datetime
from shutil import rmtree
from flask_socketio import SocketIO, emit, join_room
from flask_cors import CORS
from pymongo import MongoClient
from Configuration.config import setup_logger
from Configuration import mongo
from ultralytics import YOLOv10
from MyYolo10 import MyYolo
 

# ===============================     Server Configuration      ==================================  #

serverLog= setup_logger('server', f'./')
serverLog.info('Server has boot started')


app = Flask(__name__, static_folder='static')
app.config.from_object(mongo)
client = MongoClient(app.config['MONGO_URI'])
db = client.get_database('VideoAnalyze')
dbPrompts = db['data']

CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")  # Enable CORS for SocketIO

model = YOLOv10(f'./Configuration/weights/yolov10s.pt')
updated_prompt_workspace = {}

mainDir='./static/Data'
defaultDir= './static/Data/blank.png'

tasks = {}

serverLog.info('Server is ready')

# ===============================             Routers           ==================================  #


@app.route('/')
def index():
    global mainDir, defaultDir
    cards=[]
    try:
        files = os.listdir('./static/Data/')
        dataCount= len(files)-1
        random_files = random.sample(files, 9) if dataCount > 8 else files
        if 'blank.png' in random_files:
            random_files.remove('blank.png')
        else: 
            random_files.pop()
        

        i = 0
        while i < len(random_files):
            topicData= random_files[i].split('.')[0]
            description='' #TODO
            outDir =f'{mainDir}/{random_files[i]}/output' 
            dir=f'{mainDir}/{random_files[i]}/frames/frames'
            frame=int(len(os.listdir(dir))/2)
            dir = f'{dir}/frame_{frame}.jpg' if frame > 0 else defaultDir
            topic = topicData if frame > 0 else 'Soon'
            description = description if frame > 0 else 'Soon'
            processing = False if any(os.scandir(outDir)) else True
            cards.append([dir, topic, description,processing])
            i+=1
        while i < 8:
            cards.append([defaultDir,'Soon', ''])
            i+=1
    except:
        return render_template('index.html', cards=[])
    return render_template('index.html', cards= cards)

@app.route('/upload', methods=['POST'])
def upload():
    global serverLog, updated_prompt_workspace, tasks, model
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    formFileName = request.form.get('nameTB')
    
    serverLog.info('Server is ready')
    now = datetime.now()

    # Format the date and time
    formatted_time = now.strftime("%d-%m-%y_%H_%M")
    splittedFileName=(file.filename).split('.')
    fileName = formFileName.replace('_','---')
    serverLog.info(f"Start task - {fileName}")
    format = f'.{splittedFileName[-1]}'
    fileName= f'{fileName}_{formatted_time}'


    # Hash comparison
    file_content = file.read()
    hash = hashlib.sha1(file_content).hexdigest()
    exist ,historyFile = searchHash(hash)
    if exist:
        serverLog.info(f"Already analyzed - {historyFile}")
        return jsonify({'fileName': historyFile, 'message': 'Exist in server','status': 200})
    
    # Set up directories for analyzing
    try:
        os.makedirs(f'./static/Data/{fileName}/frames/frames')
        os.makedirs(f'./static/Data/{fileName}/frames/taggedframes')
        os.makedirs(f'./static/Data/{fileName}/input')
        os.makedirs(f'./static/Data/{fileName}/output')
        taskDir= f"./static/Data/{fileName}"
        #file.save(f'Data/{fileName}/input/{fileName}.{splittedFileName[-1]}')
        with open(f'./static/Data/{fileName}/input/{fileName}.{splittedFileName[-1]}', 'wb') as f:
            f.write(file_content)
        tasks[fileName] = MyYolo(model, taskDir, f"{fileName}{format}", serverLog, hash)
        
    except:
        dir_path = os.path.join('./static/Data', fileName)
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            rmtree(dir_path)
        return jsonify({'message': 'File Upload ERROR','status': 500})

    tasks[fileName].logSet(setup_logger(fileName, f'./static/Data/{fileName}'))
    tasks[fileName].log.info('Tree directory created')
    tasks[fileName].log.info('File uploaded')

    return jsonify({'fileName': fileName, 'format':format ,'message': 'File uploaded successfully','status': 200})

@app.route('/analyze', methods=['POST'])
def analyze():
    global tasks, serverLog
    data = request.get_json()
    fileName=data['fileName']
    try:
        task=tasks[fileName]
        task.start()
    except Exception as e:
        serverLog.error(f'An error occurred during image detection - {e}')
        close_logger(task.log)
        rmtree(task.basePath)
        return jsonify({'message': 'Video combine Error','status': 500})

    if task.updateDB(updateDBPrompt):
         task.log.info('Task has written to DB successfully')
    else:
         task.log.error('Error while writing to DB')

    serverLog.info(f"Task analyzed end successfully - {fileName}")
    return jsonify({'message': 'File uploaded successfully','status': 200, 'prompt':task.prompt})

@app.route('/res' , methods=['GET'])
def res():
    global tasks
    video_name = request.args.get('videoname')
    if not video_name:
        return render_template('notFound.html')
    decodedName = decode64(video_name)

    if not decodedName:
        return render_template('notFound.html')
    
    if decodedName in tasks:
        det = tasks[decodedName].getResDetails()
        framesPath = f'./static/Data/{det['nameNdate']}/frames/frames/'
        return render_template('res.html',videoName=det['name'],objects=det['objects'],prompts =det['prompts'],pathOrigin=det['inDir'],pathDetected=det['outDir'],statistics=det['stats'],fps=det['fps'],framesPath=framesPath)
    
    data=getVideoDetails({"name":decodedName})

    if not data:
        return render_template('notFound.html')

    format= data['videoFormat']
    input_video_path = f'./static/Data/{decodedName}/input/{decodedName}.{format}'
    tagged_video_path = f'./static/Data/{decodedName}/output/tagged_{decodedName}.{format}'
    name =decodedName.split('_')[0]
    stats=data["statistics"]
    fps= data['fps']
    framesPath = f'./static/Data/{decodedName}/frames/frames/'

    return render_template('res.html', videoName=name, objects=data['promptDistinct'], pathOrigin=input_video_path, pathDetected=tagged_video_path, statistics=stats, fps=fps, framesPath=framesPath)

@app.route('/processing')
def processing():
    return render_template('processing.html')

@app.route('/advanced')
def advanced():
    return render_template('advanced.html')

@app.route('/error')
def error():
    return render_template('error.html')

@socketio.on('updateRequest')
def on_join(data):
    global tasks
    fileName = data['fileName']
    task=tasks[fileName]
    join_room(fileName)
    if 'current' in task.prompt:
        socketio.emit(f'update@{fileName}', {"obj":task.prompt['current'], "nextStep": task.nextStep ,'status': task.processing }, room=fileName)

@app.route('/search', methods=['GET'])
def search():
    result=[]
    query = request.args.get('q', '').lower()
    if not query:
        data = getPromptsDB({})
    else:
        data = getPromptsDB( {"promptDistinct" : query} )
    values = {}
    objects=[]
    formats = []
    vid={}
    for res in data:
        outDir =f'./static/Data/{res['name']}/output' 
        vid['videoName'] = res['name'].split('_')[0]
        vid['fileName'] = res['name']
        frame=int(res['frameCount'])//2
        vid['frameSrc']= f"./static/Data/{res['name']}/frames/frames/frame_{frame}.jpg"
        vid['processing'] = False if any(os.scandir(outDir)) else True
        result.append(vid)
        objects = objects + res['promptDistinct']
        formats.append(res['videoFormat'])
        vid={}
    if not result:
        pipe=generateRegexObjectPipe(query)
        objectSuggest= getObjectFromAggregate(pipe)
        return render_template('search.html', status=404, message="Not Found", results=result, query=query,values={}, objectSuggest=objectSuggest)
    
    values["objects"] = list(set(objects))
    values["formats"] = list(set(formats))
    return render_template('search.html', status=200, message="Search Results", results=result, query=query,values=values)

@app.route('/searchWithFilters', methods=['POST'])
def searchWithFilters():
    query = {}
    data = request.get_json()
    orFilters = []
    andFilters= []


    if data['objects']:
        query['promptDistinct'] = { '$all': data['objects'] }
        for obj in data['objectFilters']:
            for filter,value in data['objectFilters'][obj].items():
                if filter == 'minSeq':
                    andFilters.append({"$expr": {"$gte": [{"$ifNull": [{"$divide": [{ "$toDouble": f"$statistics.object.{obj}.maxSequence.count" },{ "$toDouble": "$fps" }]},0]},float(value)]}})
                elif filter == 'ObjectInFrame':
                    pipe = generateObjectInFramePipe(obj,int(value))
                    andFilters.append({ "_id": { "$in": getIdFromAggregate(pipe) }})

    if data['formats']:  # More Pythonic way to check if the list is non-empty
        orFilters = [{ 'videoFormat': format } for format in data['formats']]  # Use list comprehension

    if data['#Objects']:
        andFilters.append({ "$expr": { "$gte": [{ "$size": "$promptDistinct" }, int(data['#Objects'][0])] } })



    if orFilters:
        query['$or'] = orFilters

    if andFilters:
        query['$and'] = andFilters
    


    queryRes = getPromptsDB(query)
    result=[]
    vid={}
    for res in queryRes:
        vid['videoName'] = res['name'].split('_')[0]
        vid['fileName'] = res['name']
        frame=int(res['frameCount'])//2
        vid['frameSrc']= f"./static/Data/{res['name']}/frames/frames/frame_{frame}.jpg"
        result.append(vid)
        vid={}
    if queryRes:
        return jsonify({'status': 200, 'data':result})
    else:
        return jsonify({'status': 404, 'data':[]})

        
@app.route('/getLogs', methods=['GET'])
def getLogs():
    log_file_path = os.path.join(os.path.dirname(__file__), "server.log")  # Get the path to server.log

    try:
        if os.path.exists(log_file_path):
            with open(log_file_path, "r") as log_file:
                log_content = log_file.read()
            return Response(log_content, mimetype="text/plain")  # Deliver as plain text
        else:
            return Response("Log file not found.", status=404, mimetype="text/plain")
    except Exception as e:
        return Response(f"Error reading log file: {str(e)}", status=500, mimetype="text/plain")




# ===============================        DataBase Queries       ==================================  #

def getPromptsDB(query):
    data = dbPrompts.find(query)
    result = []
    for prompt in data:
        prompt.pop('_id', None)  # Remove the '_id' field
        result.append(prompt)
    return result

def getIdFromAggregate(pipe):
    result = list(dbPrompts.aggregate(pipe))
    ids = [doc['_id'] for doc in result]
    return ids

def getObjectFromAggregate(pipe):
    result = list(dbPrompts.aggregate(pipe))
    objs = [item for doc in result for item in doc['matchedObject']]
    return list(set(objs))

def generateRegexObjectPipe(obj):
    return [
    {
        "$match": {
        "promptDistinct": {
            "$elemMatch": {
            "$regex": f"{obj}?",
            "$options": "i"
            }
        }
        }
    },
    {
        "$project": {
        "matchedObject": {
            "$filter": {
            "input": "$promptDistinct",
            "as": "item",
            "cond": { "$regexMatch": { "input": "$$item", "regex": f"{obj}?", "options": "i" } }
            }
        }
        }
    }
    ]
def generateObjectInFramePipe(obj,val):
    return[
    {
        "$project": {
            "_id": 1,
            "maxObjCount": {
                "$max": {
                    "$map": {
                        "input": { "$objectToArray": "$statistics.frame.rawData" },
                        "as": "frame",
                        "in": {
                            "$cond": {
                                "if": { "$gt": [ { "$ifNull": [ f"$$frame.v.{obj}", 0 ] }, 0 ] },
                                "then": f"$$frame.v.{obj}",
                                "else": 0
                            }
                        }
                    }
                }
            }
        }
    },
    {
        "$match": {
            "maxObjCount": { "$gte": val }
        }
    },
    {
        "$project": {
            "_id": 1  # Only return the _id field
        }
    }
]

def getVideoDetails(query):
    data = dbPrompts.find_one(query)
    if data:
        data.pop('_id', None)  # Remove the '_id' field
    return data

def updateDBPrompt(item):
    return dbPrompts.insert_one(item).acknowledged

def searchHash(hash):
    data = dbPrompts.find_one({'hash256': hash}, {'name': 1})
    if data:
        return True, data['name']
    return False, ''


# ===============================        Helpers Functions      ==================================  #

def decode64(param):
    try:
        decode= base64.b64decode(param).decode('utf-8')
        return decode
    except:
        return False
    
def close_logger(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
