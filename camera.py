import cv2
import threading

class RecordingThread (threading.Thread):
    def __init__(self, name, camera):
        threading.Thread.__init__(self)
        self.name = name 
        self.isRunning = True
        self.cap = camera 
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
        self.out = cv2.VideoWriter('/Users/jagrutimohanty/flask-sign-app-may15/static/video.mp4',fourcc,self.fps, (self.width , self.height))

    def run(self):
        while self.isRunning:
            ret, frame = self.cap.read()
            if ret:
                self.out.write(frame)
        
        self.out.release()

    def stop(self):
        self.isRunning = False
    
    def start(self):
        self.isRunning = True

    def __del__(self): 

        self.out.release()

class VideoCamera(object):
    def __init__(self):
        # Open a camera
        self.cap = cv2.VideoCapture(0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
       
      
        # Initialize video recording environment
        self.is_record = False
        self.out = None

        # Thread for recording
        self.recordingThread = None
    
    def __del__(self):
        self.cap.release()
    
    def get_frame(self):
        ret, frame = self.cap.read()
      
        if ret:
            ret, jpeg = cv2.imencode('.jpg', frame)

            #Record video
            if self.is_record:
                 if self.out == None:
                     self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

                     self.height =int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                     self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                     fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
                     self.out = cv2.VideoWriter('/Users/jagrutimohanty/flask-sign-app-may15/static/video.mp4',fourcc, self.fps, (self.width , self.height))
                
                 ret, frame = self.cap.read()
                 if ret:
                     self.out.write(frame)
            else:
                 if self.out != None:
                     self.out.release()
                     self.out = None  

            #return jpeg.tobytes() 
            return  jpeg.tobytes() 
      
        else:
            return None

    def start_record(self):
        self.is_record = True
        self.recordingThread = RecordingThread("Video Recording Thread", self.cap)
        self.recordingThread.start()

    def stop_record(self):
        self.is_record = False

        if self.recordingThread != None:
           self.recordingThread.stop()

            