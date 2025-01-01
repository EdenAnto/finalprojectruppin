import logging, os
from logging.handlers import RotatingFileHandler

MONGO_URI = "mongodb+srv://Office123:XVc2ElprezpiLIOZ@retaurantoffice.hdnt4lz.mongodb.net/"

class VideoDetectionCls:
    def __init__(self, videoName, fileName, format, hash):
        self.fileName = fileName
        self.name = videoName
        self.basePath = f'./static/Data/{self.fileName}'
        self.framesPath = f'./static/Data/{self.fileName}/frames/frames'
        self.taggedFramesPath = f'./static/Data/{self.fileName}/frames/taggedframes'
        self.inputPath = f'./static/Data/{self.fileName}/input/{fileName}{format}'
        self.outputFolder = f'./static/Data/{self.fileName}/output'
        self.outputPath = f'{self.outputFolder}/tagged_{self.fileName}{format}'
        self.hash=hash
        self.format=format
        self.frameCount=0
        self.data={}
        self.prompt={'current': ''}
        self.promptSet=set()
        self.nextStep= [-1]
        self.processing=''

        
    def logSet(self, logger):
        self.log = logger
    
    def updateProcessing(self, update):
        self.processing = update
    
        

def setup_logger(name, log_file, level=logging.INFO):
    logPath = f'{log_file}/{name}.log'
    os.makedirs(os.path.dirname(logPath), exist_ok=True)

    handler = RotatingFileHandler(logPath, maxBytes=1024*1024*10, backupCount=5)
    handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s \t %(levelname)s: %(message)s \t [in %(pathname)s:%(lineno)d]')
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers to avoid duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.addHandler(handler)
    logger.propagate = False  # Prevent log messages from being propagated to the root logger
    
    return logger


