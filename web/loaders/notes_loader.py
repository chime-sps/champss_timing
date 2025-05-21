import datetime
from ..services.run_notes import RunNotes

class notes_loader():
    def __init__(self, notebook_path, app):#, auto_update = False):
        self.app = app
        self.notebook_path = notebook_path
        
    def fetch(self, user=None, tag=None, time_range=None, sort=True, group_by_date=True):
        # Load notes
        notes = []
        with RunNotes(self.notebook_path, readonly=True) as notes:
            notes = notes.read(user=user, tag=tag, time_range=time_range, sort=sort)

        # Parse notes
        for i, note in enumerate(notes):
            note["date"] = datetime.datetime.fromtimestamp(note["timestamp"]).strftime('%Y-%m-%d (%a)')
            note["time"] = datetime.datetime.fromtimestamp(note["timestamp"]).strftime('%H:%M:%S')

        # Return notes if no date grouping is needed
        if not group_by_date:
            return notes

        # Gruop notes by date
        notes_dated = {}
        for note in notes:
            date = note["date"]
            if date not in notes_dated:
                notes_dated[date] = []
            notes_dated[date].append(note)

        return notes_dated