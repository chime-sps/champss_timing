import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from .utils import utils

class slack_handler:
    def __init__(self, slack_token):
        self.SLACK_BOT_TOKEN = slack_token["SLACK_BOT_TOKEN"]
        self.SLACK_APP_TOKEN = slack_token["SLACK_APP_TOKEN"]
        self.CHANNEL_ID = slack_token["CHANNEL_ID"]
        self.app = App(token=self.SLACK_BOT_TOKEN)

    def send(self, message=None, image=None, file=None, image_title=None, file_title=None):
        if message is not None:
            self.send_text(message)
        if image is not None:
            if image_title is not None:
                self.send_file(image, image_title)
            else:
                self.send_file(image)
        if file is not None:
            if file_title is not None:
                self.send_file(file, file_title)
            else:
                self.send_file(file)
    
    def send_text(self, message):
        try:
            self.app.client.chat_postMessage(channel=self.CHANNEL_ID, text=message)
        except Exception as e:
            utils.print_error(f"Error sending message: {e}")
            utils.print_error(f"Message: {message}")

    def send_file(self, filename, message=""):
        try:
            self.app.client.files_upload_v2(
                channels=self.CHANNEL_ID,
                initial_comment=message,
                file=filename,
                filename=filename
            )
        except Exception as e:
            utils.print_error(f"Error sending file: {e}")
            utils.print_error(f"File: {filename}")

class print_handler:
    def __init__(self):
        return

    def send(self, message=None, image=None, file=None, image_title=None, file_title=None):
        if message is not None:
            self.print_text(message)
        if image is not None:
            self.print_text("Image: " + image)
        if file is not None:
            self.print_text("File: " + file)

    def print_text(self, message):
        print("\033[1;32;40m(Alert -> " + message + ")\033[0m")
        
class notification:
    def __init__(self, messager_token):
        if not messager_token:
            self.sh = print_handler()
        else:
            self.sh = slack_handler(messager_token)

    def send_urgent_message(self, message, psr_id="psr_id_not_provided"):
        self.sh.send("[ ⚠️ URGENT ! ] <@wenke.xia>\n" + message + "\nPSR ID: #" + psr_id)

    def send_success_message(self, message, psr_id="psr_id_not_provided"):
        self.sh.send("[ ✅ SUCCESS ]\n" + message + "\nPSR ID: #" + psr_id)

    def send_message(self, message, psr_id="psr_id_not_provided"):
        self.sh.send(message + "\nPSR ID: #" + psr_id)

    def send_image(self, image, psr_id="psr_id_not_provided"):
        self.sh.send(image=image, image_title="PSR ID: #" + psr_id)

    def send_file(self, file, psr_id="psr_id_not_provided"):
        self.sh.send(file=file, file_title="PSR ID: #" + psr_id)
    
    def send_code(self, code, psr_id="psr_id_not_provided"):
        self.sh.send(message=f"```\n{code}\n```" + "\nPSR ID: #" + psr_id)

# sh = slack_handler()
# sh.send("Hello, World!")
# sh.send(image="/Users/wenky/Downloads/champss_diagnostic.jpg")
# sh.send(file="/Users/wenky/Downloads/champss_diagnostic.pdf")