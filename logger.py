import time
import copy

from .notification import notification

class logger():
    def __init__(self, psr_id="NA", noti=False, level="INFO"):
        self.psr_id = psr_id
        self.level = level
        self.default_layer = 0
        self.notification = noti
        self.noti_hdl = notification()
    
    def copy(self):
        return logger(psr_id=self.psr_id, noti=self.notification, level=self.level)

    def get_time_string(self):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    
    def format_text(self, text, level, layer, color=None):
        output = ""
        
        output += "  " * layer
        if layer > 0:
            output += "│"

        output += f"{level} {text}"

        if color is not None:
            if color == "red":
                output = f"\033[91m{output}\033[0m"
            elif color == "green":
                output = f"\033[92m{output}\033[0m"
            elif color == "yellow":
                output = f"\033[93m{output}\033[0m"
            elif color == "blue":
                output = f"\033[94m{output}\033[0m"
        
        output = f"{self.get_time_string()} {output}"

        return output
    
    def info(self, text, layer=0, end="\n"):
        if layer == 0:
            layer = self.default_layer

        print(self.format_text(text, "INFO   ", layer, color="blue"), end=end)

    def warning(self, text, layer=0, end="\n"):
        if layer == 0:
            layer = self.default_layer

        print(self.format_text(text, "WARNING", layer, color="yellow"), end=end)
        
        if self.notification:
            self.noti_hdl.send_message(text, psr_id=self.psr_id)

    def error(self, text, layer=0, end="\n"):
        if layer == 0:
            layer = self.default_layer
            
        print(self.format_text(text, "ERROR  ", layer, color="red"), end=end)
        
        if self.notification:
            self.noti_hdl.send_urgent_message(text, psr_id=self.psr_id)

    def success(self, text, layer=0, end="\n"):
        if layer == 0:
            layer = self.default_layer

        print(self.format_text(text, "SUCCESS", layer, color="green"), end=end)

    def debug(self, text, layer=0, end="\n"):
        if layer == 0:
            layer = self.default_layer

        print(self.format_text(text, "DEBUG  ", layer), end=end)