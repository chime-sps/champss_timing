import time
import copy

# from .notification import notification

class logger():
    def __init__(self, noti=False, level="INFO"):
        self.level = level
        self.default_layer = 0
        # self.notification = noti
        # self.noti_hdl = notification()
    
    def copy(self, level_up=True):
        logger = copy.deepcopy(self)
        
        if level_up:
            logger.level_up()

        return logger
    
    def level_up(self):
        self.default_layer += 1

    def level_down(self):
        self.default_layer -= 1

    def get_time_string(self):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    
    def format_text(self, text, level, layer, color=None):
        output = ""
        
        output += "  " * int(layer)
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
        print(self.format_text(text, "INFO   ", self.default_layer + layer, color="blue"), end=end)

    def warning(self, text, layer=0, end="\n"):
        print(self.format_text(text, "WARNING", self.default_layer + layer, color="yellow"), end=end)
        
        # if self.notification:
        #     self.noti_hdl.send_message(text, psr_id=self.psr_id)

    def error(self, text, layer=0, end="\n"):
        try:
            print(self.format_text(text, "ERROR  ", self.default_layer + layer, color="red"), end=end)
        except Exception as e:
            print("ERROR: ", text)
            print("\033[91m[ Logger Error ] While printing the error message, an error occurred: ", e, "\033[0m")
        
        # if self.notification:
        #     self.noti_hdl.send_urgent_message(text, psr_id=self.psr_id)

    def success(self, text, layer=0, end="\n"):
        print(self.format_text(text, "SUCCESS", self.default_layer + layer, color="green"), end=end)

    def debug(self, text, layer=0, end="\n"):
        print(self.format_text(text, "DEBUG  ", self.default_layer + layer), end=end)