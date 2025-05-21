import os
import time
import json
import sqlite3
import threading
from queue import Queue
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from backend.utils.logger import logger
from backend.utils.utils import utils

class RunNotes:
    def __init__(self, notebook_path, readonly=False, check_same_thread=True, logger=logger()):
        # Define variables
        self.notebook_path = notebook_path
        self.check_same_thread = check_same_thread
        self.readonly = readonly
        self.logger = logger

        # Make sure the database path exists
        if not os.path.exists(os.path.dirname(self.notebook_path)):
            raise ValueError(f"Database path {self.notebook_path} does not exist.")

        # Connect to the database
        if self.readonly:
            # check if database exists
            if not os.path.exists(notebook_path):
                raise Exception(f"Database {notebook_path} does not exist. Please provide a valid database file.")

            # copy the database to a temporary file
            self.notebook_path = os.path.abspath(f"{notebook_path}.readonly{utils.get_rand_string()}.tmp")
            shutil.copyfile(notebook_path, self.notebook_path)
            self.logger.debug(f"Readonly temporary readonly notebook created at {self.notebook_path}")

            # open the temporary database in readonly mode
            self.conn = sqlite3.connect("file://" + self.notebook_path + "?mode=ro", uri=True, check_same_thread=False)
        else:
            self.conn = sqlite3.connect(self.notebook_path, check_same_thread=self.check_same_thread)
        
        # Get cursor
        self.cursor = self.conn.cursor()

    def initialize(self):
        if self.readonly:
            return 

        # Create a table for storing notebook data
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS notebooks (
                id INTEGER PRIMARY KEY,
                timestamp INT,
                user TEXT,
                content LONGTEXT,
                tag TEXT, 
                metadata LONGTEXT
            )
        ''')

        # Create indexes for the table
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON notebooks (timestamp)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_user ON notebooks (user)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_tag ON notebooks (tag)')

        self.conn.commit()
    
    def __worker(self):
        # Worker thread to process the queue
        while True:
            item = self.queue.get()
            
            # Sleep if no item is available
            if item is None:
                time.sleep(1)

            # Process the item
            try:
                # Write
                if item["action"] == "write":
                    self.write(
                        content=item["content"], 
                        user=item["user"],
                        tag=item["tag"],
                        metadata=item["metadata"], 
                        force=True
                    )
                else:
                    raise ValueError(f"Unknown action {item['action']}")

                # Mark the task as done
                self.results[item["task_id"]] = True
            except Exception as e:
                print(f"Error writing notes: {e}")

                # Mark the task as failed
                self.results[item["task_id"]] = False
                
            self.queue.task_done()

    def start_worker(self):
        if self.readonly:
            raise ValueError("Cannot start worker thread in readonly mode.")

        self.queue = Queue()
        self.results = {}
        self.worker_thread = threading.Thread(target=self.__worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
    
    def queue_write(self, content, user, tag, metadata, wait=False):
        if not hasattr(self, 'queue'):
            raise ValueError("Worker thread not started. Call start_worker() first.")

        # Create the task
        task = {
            "task_id": utils.get_rand_string(),
            "action": "write",
            "content": content,
            "user": user,
            "tag": tag,
            "metadata": metadata
        }

        # Queue the write operation
        self.queue.put(task)

        if wait:
            # Wait for the queue to be processed
            while task["task_id"] not in self.results:
                time.sleep(1)

            # Check if successful
            if not self.results[task["task_id"]]:
                return False

        return True
            
        
    def stop_worker(self):
        # Stop the worker thread
        self.queue.put(None)
        self.worker_thread.join()
        self.queue.join()

    def get_tags(self):
        # Get all unique tags from the database
        self.cursor.execute('SELECT DISTINCT tag FROM notebooks')
        tags = [row[0] for row in self.cursor.fetchall()]

        # Sanity check
        if not tags:
            tags = []
        
        # Add the "main" tag if it doesn't exist
        if "main" not in tags:
            tags.append("main")

        return tags

    def write(self, content, user, tag, metadata, force=False):
        if self.readonly:
            raise ValueError("Writing to a readonly database is not allowed.")

        if not force and not self.check_same_thread:
            raise ValueError("Database connection is not thread-safe. Use queue_write() instead.")

        # Sanity check the tag
        if "\"" in tag or "'" in tag:
            raise ValueError("Tag cannot contain \" or ' character. Please use a valid tag.")
            
        # Write data to the database
        self.cursor.execute('''
            INSERT INTO notebooks (timestamp, user, content, tag, metadata)
            VALUES (?, ?, ?, ?, ?)
        ''', (int(time.time()), user, content, tag, json.dumps(metadata)))

        self.conn.commit()

    def format(self, notes):
        # Format notes for display
        formatted_notes = []
        for note in notes:
            formatted_note = {
                "id": note[0],
                "timestamp": note[1],
                "user": note[2],
                "content": note[3],
                "tag": note[4],
                "metadata": json.loads(note[5])
            }
            formatted_notes.append(formatted_note)
        return formatted_notes

    def read(self, user=None, tag=None, time_range=None):
        # Read data from the database
        query = 'SELECT * FROM notebooks WHERE 1=1'
        params = []

        if user:
            query += ' AND user = ?'
            params.append(user)

        if tag:
            query += ' AND tag = ?'
            params.append(tag)

        if time_range:
            query += ' AND timestamp BETWEEN ? AND ?'
            params.extend(time_range)

        self.cursor.execute(query, params)
        return self.format(self.cursor.fetchall())

    def close(self):
        # Close the database connection
        self.conn.close()

    def __enter__(self):
        # Initialize the database
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close the database connection
        self.close()

        # Stop the worker thread
        if hasattr(self, 'worker_thread'):
            self.stop_worker()

class SlackRunNotes:
    def __init__(self, notebook_path, slack_token, psrs=[]):
        # Define variables
        self.notebook = RunNotes(notebook_path, check_same_thread=False)
        self.SLACK_BOT_TOKEN = slack_token["SLACK_BOT_TOKEN"]
        self.SLACK_APP_TOKEN = slack_token["SLACK_APP_TOKEN"]
        self.psrs = psrs

        # Initialize the notebook
        self.notebook.initialize()

        # Initialize app
        self.app = App(token=self.SLACK_BOT_TOKEN)
        self.socket_mode_handler = SocketModeHandler(self.app, self.SLACK_APP_TOKEN)

        # Define handlers
        @self.app.event("app_mention")
        def handle_app_mention_events(body, logger, say):
            self.handle_app_mention_events(body, logger, say)

    def start(self):
        # Start the worker thread
        self.notebook.start_worker()

        # Start the app
        self.socket_mode_handler.start()
    
    def show_help(self):
        return f"Available commands:\n - `@bot [post (=\"main\") or <tag_name>] <content>`: Post a new note with the specified tag.\n - `@bot help`: Show this help message."

    def handle_app_mention_events(self, body, logger, say):
        # Retrieve thread timestamp or fallback to current message timestamp
        thread_ts = body["event"].get("thread_ts", body["event"]["ts"])

        # Identify the user who sent the message
        user_id = body["event"]["user"]
        user_screenname = self.app.client.users_info(user=user_id)["user"]["real_name"]

        # Get the message text
        msg = body["event"]["text"].split(" ")
        if len(msg) < 2:
            say(f"<@{user_id}> Sorry, I didn't understand that. Please use `@bot help` to see available commands.",
                thread_ts=thread_ts)
            return

        # Parse the message
        action = msg[1].strip()
        args = msg[2:]

        # Replace action with "main" if it's "post"
        if action == "post":
            action = "main" # Posting to timing tag by default

        # Run the action
        if action == "help":
            # Send help message
            say(f"<@{user_id}> {self.show_help()}", thread_ts=thread_ts)
        else:
            # Sanity check
            if len(args) < 1:
                say(f"<@{user_id}> Usage: @bot <tag_name> <content>", thread_ts=thread_ts)
                return

            # Create the content etc.
            content = " ".join(args)
            tag = action
            metadata = {}

            # Parse the "tag" notation
            confirm_tag = False
            tag = tag.replace("'", "\"").replace("”", "\"").replace("“", "\"")
            if "\"" in tag:
                if tag.startswith("\"") and tag.endswith("\""):
                    confirm_tag = True

                # Remove the * from the tag 
                tag = tag.replace("\"", "")

            # Check if this action will create a new tag
            if tag not in self.notebook.get_tags() and tag not in self.psrs:
                if not confirm_tag:
                    say(f"<@{user_id}> This is looks like the first note under the `{tag}` tag.\n - If you want to create this tag while posting, please use `@bot \"{tag}\" {content}` instead. \n - If {tag} is meant to be a comment, please use `@bot help` to see all available commands.", thread_ts=thread_ts)
                    return

            # Insert the note into the database
            try:
                if not self.notebook.queue_write(content, user_screenname, tag, metadata, wait=True):
                    say(f"<@{user_id}> Error writing note, please try again. If the problem persists, please create an issue on GitHub.", thread_ts=thread_ts)
                    return
                say(f"<@{user_id}> Note added: \n```[{tag}] @{user_screenname}: {content}```\n Thanks for keeping the pipeline running 🎉", thread_ts=thread_ts)
            except Exception as e:
                say(f"<@{user_id}> Error adding note: {e}", thread_ts=thread_ts)
        

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close the database connection
        self.notebook.close()