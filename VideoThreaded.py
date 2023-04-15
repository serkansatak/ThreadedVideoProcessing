import cv2 
from typing import Union, Callable
from multiprocessing.pool import ThreadPool
from collections import deque
from collections.abc import Iterable, Callable
import os
from time import time
import numpy as np

class StatValue:
    def __init__(self, smooth_coef = 0.5):
        self.value = None
        self.smooth_coef = smooth_coef
        self.valueList = []
        
    def update(self, v):
        if self.value is None:
            self.value = v
        else:
            c = self.smooth_coef
            self.value = c * self.value + (1.0-c) * v
        self.valueList.append(self.value)
        
def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()

class VideoProcessor :
    def __init__(self, 
                 src: Union[str, int], 
                 out: str = None, 
                 benchmark: bool = False, 
                 outsize : tuple[int, int] = None,
                 single_thread: bool = False):
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
        self.single_thread = single_thread
        if self.single_thread:
            self.pool = ThreadPool(processes=1)
        else:
            self.pool = ThreadPool(processes=self.threadNum)
        self.pending = deque()
        self.benchmark = benchmark
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frameNum = 0
        
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
        t0 = time()
        self.update()
        execTime = time()-t0
        print(f"Execution time : {execTime:.2f} seconds.")
        print(f"Time per frame : {(execTime*1000/self.frameCount):.1f} ms.")
        if self.benchmark:
            self.writeExecTimeInfo()
        
    def update(self):
        t = 0
        ret = 1
        
        while self.cap.isOpened():
            
            while len(self.pending) > 0 and self.pending[0].ready():
                res, t0 = self.pending.popleft().get()
                self.frameNum += 1
                if self.benchmark:
                    self.latency.update(clock()-t0)     
                if self.out:
                    self.outWriter.write(res)
            
            if self.frameNum == self.frameCount:
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
        elif isinstance(value, Callable):
            self._operator = value
            self._operatorArgs = {}
        else:
            raise Exception("You should send a function or a dictionary which containse function and its kwargs.")    
    
    def printAttributes(self):
        for key,value in self.__dict__.items():
            print(f"{key} : {value}")
            
    def writeExecTimeInfo(self):
        if self.benchmark:
            with open(f"{self.__class__}_exec.info", "w") as f:
                f.write(f"---- Latency ---- Average: \
                    {(sum(self.latency.valueList*1000)/len(self.latency.valueList)):.1f} ms\n")
                for idx, val in enumerate(self.latency.valueList, start=1):
                    f.write(f"FrameNo: {idx} -- Latency: {(val*1000):.1f} ms\n")
                
                f.write(f"\n\n************************************\n\n")
                f.write(f"---- Frame Interval ---- Average: \
                    {(sum(self.frame_interval.valueList*1000)/len(self.frame_interval.valueList)):.1f} ms\n")
                for idx, val in enumerate(self.frame_interval.valueList, start=1):
                    f.write(f"FrameNo: {idx} -- Frame Interval: {(val*1000):.1f} ms\n")
                    f.close()


class SequenceProcessor :
    def __init__(self, 
                 src: str, 
                 imageExt: str = ".png",
                 out: str = "video.mp4", 
                 outsize : tuple[int, int] = None,
                 fps : int = None,
                 benchmark: bool = False, 
                 single_thread: bool = False
                 ):
        """
        src : str for videofile, int for webcam
        out : path for output file
        benchmark : print execution times or not
        
        For output files only .mp4 format is available with 'MP4V' fourcc right now.
        """
        
        import glob
                
        self.writerSet = False
                
        self.src = src
        self.imageExt = imageExt
        self.out = os.path.join(self.src, out) if os.path.split(out)[0] == None else out
                
        self.toVid = True if os.path.splitext(self.out)[-1] else False
            
        self.outsize = outsize
        self.threadNum = cv2.getNumberOfCPUs()
        self.single_thread = single_thread
        if self.single_thread:
            self.pool = ThreadPool(processes=1)
        else:
            self.pool = ThreadPool(processes=self.threadNum)
        self.pending = deque()
        self.benchmark = benchmark
        self.fps = fps if fps else 24
        self.frameNum = 0
        self.imageList = sorted(glob.glob(os.path.join(self.src, f"*{self.imageExt}")))
        self.frameCount = len(self.imageList)
        
        if not self.toVid:
            os.makedirs(self.out, exist_ok=True)
        
        if self.benchmark:
            self.latency = StatValue()
            self.frame_interval = StatValue()
            self.last_frame_time = clock()
            
    def processSequence(self):
        t0 = time()
        self.update()
        execTime = time()-t0
        print(f"Execution time : {execTime:.2f} seconds.")
        print(f"Time per frame : {(execTime*1000/self.frameCount):.1f} ms.")
        if self.benchmark:
            self.writeExecTimeInfo()
            
    def write(self, frame, frameNum):
        if self.toVid:
            self.writer.write(frame)
        else:
            self.writer(os.path.join(self.out,f"{frameNum}.png"), frame)    
            
    @property
    def writer(self):
        return self._writer
    
    @writer.setter
    def writer(self, val):
        self._writer = val
        self.writerSet = True

    def update(self):
        t = 0
        
        while self.frameNum < self.frameCount+1:
            
            while len(self.pending) > 0 and self.pending[0].ready():
                res, t0, frameNum = self.pending.popleft().get()
                if self.benchmark:
                    self.latency.update(clock()-t0)     
                if self.out:
                    self.write(res, frameNum)
            
            if self.frameNum >= self.frameCount:
                break
                
            if len(self.pending) < self.threadNum:
                frame = cv2.imread(self.imageList[self.frameNum])
                self.frameNum += 1
                
                if not self.writerSet:
                    if self.toVid:
                        self.outsize = frame.shape[:2]
                        self.writer = cv2.VideoWriter(self.out, cv2.VideoWriter_fourcc(*'mp4v'), float(self.fps), self.outsize)
                    else:
                        self.writer = cv2.imwrite
                                
                if self.benchmark:
                    t = clock()
                    self.frame_interval.update(t-self.last_frame_time)
                    self.last_frame_time = t
                
                task = self.pool.apply_async(self.processFrame, (frame, t, self.frameNum))
                self.pending.append(task)          
                            
    def processFrame(self, frame, t0, frameNum):
        a = []
        result = self.operator(frame, t0, **self._operatorArgs)
        if isinstance(result, Iterable):
            a.extend(self.operator(frame, t0, **self._operatorArgs))
        else:
            a.append(result)
        a.append(frameNum)
        return a

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
        elif isinstance(value, Callable):
            self._operator = value
            self._operatorArgs = {}
        else:
            raise Exception("You should send a function or a dictionary which containse function and its kwargs.")    
    
    def printAttributes(self):
        for key,value in self.__dict__.items():
            if key == "images":
                continue
            if isinstance(value, np.ndarray):
                continue
            print(f"{key} : {value}")
            
            
    def writeExecTimeInfo(self):
        if self.benchmark:
            with open(f"{self.__class__}_exec.info", "w") as f:
                f.write(f"---- Latency ---- Average: \
                    {(sum(self.latency.valueList*1000)/len(self.latency.valueList)):.1f} ms\n")
                for idx, val in enumerate(self.latency.valueList, start=1):
                    f.write(f"FrameNo: {idx} -- Latency: {(val*1000):.1f} ms\n")
                
                f.write(f"\n\n************************************\n\n")
                f.write(f"---- Frame Interval ---- Average: \
                    {(sum(self.frame_interval.valueList*1000)/len(self.frame_interval.valueList)):.1f} ms\n")
                for idx, val in enumerate(self.frame_interval.valueList, start=1):
                    f.write(f"FrameNo: {idx} -- Frame Interval: {(val*1000):.1f} ms\n")
                    f.close()

if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str, help='Input video file')
    parser.add_argument("--output", type=str, default=None, help='Output file path.')
    parser.add_argument("--benchmark", action='store_true')
    parser.add_argument("--single-thread", action='store_true')
    args = parser.parse_args()
    
    def test_outer():
        processor = VideoProcessor(args.input, args.output, benchmark=args.benchmark, single_thread=args.single_thread)
        operation = lambda frame, t0, blurRate: (cv2.medianBlur(frame, blurRate), t0)
        processor.operator = {'func': operation, 'kwargs': {'blurRate': 19}}
        processor.processVideo()
        processor.printAttributes()
    
    def test_internal():
        class TestProcessor(VideoProcessor):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.counter = 0
            
            def testOperator(self, frame, t, blurRate):
                self.counter += 1
                return cv2.medianBlur(frame, blurRate), t
                
        processor = TestProcessor(src=args.input, out=args.output, benchmark=args.benchmark, single_thread=args.single_thread)
        processor.operator = {'func': processor.testOperator, 'kwargs': {'blurRate': 19}}
        processor.processVideo()
        processor.printAttributes()
        
    def test_sequence():
        class TestProcessor(SequenceProcessor):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.counter = 0
            
            def testOperator(self, frame, t, blurRate):
                self.counter += 1
                return cv2.medianBlur(frame, blurRate), t
            
        processor = TestProcessor(args.input, out=args.output, benchmark=args.benchmark, single_thread=args.single_thread)
        processor.operator = {'func': processor.testOperator, 'kwargs': {'blurRate': 19}}
        processor.processSequence()
        processor.printAttributes()
    test_sequence()
    exit()