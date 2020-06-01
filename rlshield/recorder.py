import gridstorm.plotter as plotter
import gridstorm.trace as trace

class Recorder:
    def __init__(self, finishers_only):
        self._only_keep_finishers = finishers_only

class VideoRecorder(Recorder):
    def __init__(self, renderer, only_keep_finishers):
        super().__init__(only_keep_finishers)
        self._paths = []
        self._path = None
        self._renderer = renderer

    def start_path(self):
        assert self._path is None
        self._path = trace.BeliefTrace()

    def end_path(self, finished):
        self._path.append_action(None)
        if not self._only_keep_finishers or finished:
            self._paths.append(self._path)
        self._path = None

    def record_state(self, state):
        self._path.append_state(state)

    def record_belief(self, belief):
        self._path.append_potential_states(belief)

    def record_selected_action(self, action):
        self._path.append_action(action)

    def record_available_actions(self, actions):
        self._path.append_available_actions(actions)

    def record_allowed_actions(self, actions):
        self._path.append_considered_actions(actions)

    def trim_from_end(self, length):
        for path in self._paths:
            path.trim_from_end(length)

    def save(self):
        for i, trace in enumerate(self._paths):
            mp4file = f"test-run{i}.mp4"
            self._renderer.record(mp4file, trace)


class LoggingRecorder(Recorder):
    """
    A very simple general purpose recorder

    """
    def __init__(self, only_keep_finishers):
        super().__init__(only_keep_finishers)
        self._paths = []
        self._observed_paths = []
        self._path = None
        self._observed_path = None

    def start_path(self):
        assert self._path is None
        assert self._observed_path is None
        self._path = []
        self._observed_path = []

    def end_path(self, finished):
        if not self._only_keep_finishers or finished:
            self._paths.append(self._path)
            self._observed_paths.append(self._observed_path)
        self._path = None
        self._observed_path = None

    def record_state(self, state):
        self._path.append(f"{state}")

    def record_belief(self, belief):
        self._observed_path.append(f"{belief}")

    def record_selected_action(self, action):
        self._path.append(f"--act={action}-->")

    def record_available_actions(self, actions):
        pass

    def record_allowed_actions(self, actions):
        pass

    def save(self):
        for path in self._paths:
            print(" ".join(path))
        for observed_path in self._observed_paths:
            print(" ".join(observed_path))

