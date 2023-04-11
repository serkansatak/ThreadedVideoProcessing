import cv2 
from typing import Union, Callable
from multiprocessing.pool import ThreadPool
from collections import deque
import os

class StatValue:
    def __init__(self, smooth_coef = 0.5):
        self.value = None
        self.smooth_coef = smooth_coef
    def update(self, v):
        if self.value is None:
            self.value = v
        else:
            c = self.smooth_coef
            self.value = c * self.value + (1.0-c) * v
            
def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()

class VideoProcessor :
    def __init__(self, 
                 src: Union[str, int], 
                 out: str = None, 
                 benchmark: bool = False, 
                 outsize : tuple[int, int] = None):
        """
        src : str for videofile, int for webcam
        out : path for output file
        benchmark : print execution times or not
        
        For output files only .mp4 format is available with 'MP4V' fourcc right now.
        """
        
        self.src = src
        self.out = out
        self.outsize = outsize
        self.cap = cv2.VideoCapture(self.src)
        self.threadNum = cv2.getNumberOfCPUs()
        self.pool = ThreadPool(processes=self.threadNum)
        self.pending = deque()
        self.benchmark = benchmark
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.frameCount = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        if self.out:
            self.original_fourcc = self.cap.get(cv2.CAP_PROP_FOURCC)
            outDir, outName = os.path.split(self.out)
            
            if not os.path.exists(outDir):
                os.makedirs(outDir, exist_ok=False)
            
            if outName:
                outName, outExt = os.path.splitext(outName)
            else:
                outName = 'output'
            outExt = ".mp4"
            self.out = os.path.join(outDir, outName + outExt)
            if not self.outsize:
                self.outsize = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) 
            self.outWriter = cv2.VideoWriter(self.out, cv2.VideoWriter_fourcc(*'mp4v'), float(self.fps), self.outsize)
        
        if self.benchmark:
            self.latency = StatValue()
            self.frame_interval = StatValue()
            self.last_frame_time = clock()
            
    def processVideo(self):
        self.update()
        
    def update(self):
        t = 0
        ret = 1
        frameNum = 0
        
        while self.cap.isOpened():
            
            while len(self.pending) > 0 and self.pending[0].ready():
                res, t0 = self.pending.popleft().get()
                frameNum += 1
                if self.benchmark:
                    self.latency.update(clock()-t0)     
                if self.out:
                    self.outWriter.write(res)
            
            if frameNum == self.frameCount:
                break
                
            if len(self.pending) < self.threadNum:
                ret, frame = self.cap.read()
                if ret:
                    if self.benchmark:
                        t = clock()
                        self.frame_interval.update(t-self.last_frame_time)
                        self.last_frame_time = t
                    
                    task = self.pool.apply_async(self.processFrame, (frame.copy(), t))
                    self.pending.append(task)          
                                
    def processFrame(self, frame, t0):
        return self.operator(frame, t0, **self._operatorArgs)

    @property
    def operator(self):
        return self._operator
    
    @operator.setter
    def operator(self, 
                 value: Union[dict[Callable, dict], Callable]):
        """
        value : {'func' : operator function, 'kwargs': keyword arguments}
        
        Function should take frame, t and kwargs as inputs whether or not you use them.\n
        Function should return output frame and time.
        """
        
        if isinstance(value, dict):
            self._operator = value['func']
            self._operatorArgs = value.get('kwargs', {})
        elif isinstance(value, function):
            self._operator = value
            self._operatorArgs = {}
        else:
            raise Exception("You should send a function or a dictionary which containse function and its kwargs.")    
        

if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str, help='Input video file')
    parser.add_argument("--output", type=str, default=None, help='Output file path.')
    args = parser.parse_args()
    
    
    processor = VideoProcessor(args.input, args.output)
    
    operation = lambda frame, t0, blurRate: (cv2.medianBlur(frame, blurRate), t0)
    processor.operator = {'func': operation, 'kwargs': {'blurRate': 19}}
    
    processor.processVideo()
    exit()