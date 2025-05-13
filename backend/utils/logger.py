import time
import copy
import inspect

# from .notification import notification

class logger():
    def __init__(self, noti=False, level="INFO"):
        self.level = level
        self.default_layer = 0
        self.log_cache = []
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
    
    def get_stack_info(self, max_len=30):
        stack = inspect.stack()
        stack_info = []

        for i in range(1, len(stack)):
            if stack[i].function == "<module>":
                break
            
            if "logger.py" in stack[i].filename:
                continue

            stack_info.append(stack[i].function)
        
        stack_info_string = ""
        stack_info.reverse()

        for this_stack in stack_info:
            if this_stack == "format_text":
                break
            stack_info_string += f"{this_stack}:"
        
        stack_info_string = "main:" + stack_info_string
        stack_info_string = stack_info_string[:-1]

        if len(stack_info_string) > max_len:
            stack_info_string = "..." + stack_info_string[-(max_len-3):]
        
        return stack_info_string
    
    def format_text(self, text, level, layer, color=None, marker="│", time=True, stack=True, cache_log=True):
        output = ""
        
        output += "    " * int(layer)
        # if layer > 0:
        #     output += marker + " "
        output += marker + " "

        output += f"{level} "

        if stack:
            output += f"{self.get_stack_info()}  "

        output += text

        if color is not None:
            if color == "red":
                output = f"\033[91m{output}\033[0m"
            elif color == "green":
                output = f"\033[92m{output}\033[0m"
            elif color == "yellow":
                output = f"\033[93m{output}\033[0m"
            elif color == "blue":
                output = f"\033[94m{output}\033[0m"
            elif color == "purple":
                output = f"\033[95m{output}\033[0m"
        
        if time:
            output = f"{self.get_time_string()} {output}"

        if cache_log:
            self.log_cache.append(output)

        return output

    def info(self, *args, layer=0, end="\n"):
        text = " ".join([str(arg) for arg in args])

        for line in text.split("\n"):
            print(self.format_text(line, "INFO   ", self.default_layer + layer, color="blue"), end=end)

    def warning(self, *args, layer=0, end="\n"):
        text = " ".join([str(arg) for arg in args])

        for line in text.split("\n"):
            print(self.format_text(line, "WARNING", self.default_layer + layer, color="yellow"), end=end)
        
        # if self.notification:
        #     self.noti_hdl.send_message(text, psr_id=self.psr_id)

    def error(self, *args, layer=0, end="\n"):
        text = " ".join([str(arg) for arg in args])

        try:
            for line in text.split("\n"):
                print(self.format_text(line, "ERROR  ", self.default_layer + layer, color="red"), end=end)
        except Exception as e:
            print("ERROR: ", text)
            print("\033[91m[ Logger Error ] While printing the error message, an error occurred: ", e, "\033[0m")
        
        # if self.notification:
        #     self.noti_hdl.send_urgent_message(text, psr_id=self.psr_id)

    def success(self, *args, layer=0, end="\n"):
        text = " ".join([str(arg) for arg in args])

        for line in text.split("\n"):
            print(self.format_text(line, "SUCCESS", self.default_layer + layer, color="green"), end=end)

    def debug(self, *args, layer=0, end="\n"):
        text = " ".join([str(arg) for arg in args])

        for line in text.split("\n"):
            print(self.format_text(line, "DEBUG  ", self.default_layer + layer), end=end)

    def data(self, *args, layer=0, end="\n"):
        text = " ".join([str(arg) for arg in args])

        for line in text.split("\n"):
            print(self.format_text(line, "       ", self.default_layer + layer, color="purple"), end=end)

    def get_log_cache(self):
        return self.log_cache

    def clear_log_cache(self):
        self.log_cache = []

    def save_log(self, file_path, clear=True):
        with open(file_path, "w") as f:
            for log in self.log_cache:
                f.write(log + "\n")

        if clear:
            self.clear_log_cache()