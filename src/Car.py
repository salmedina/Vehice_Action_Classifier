from utils.utils import calc_vector, angle_between
from collections import namedtuple

CarState = namedtuple('CarState', ['frame', 'cx', 'cy', 'vec', 'norm', 'angle', 'moving'])

class Car():
    '''This class stores a list of states of the tracked car by window sizes'''
    def __init__(self):
        self.states = {}

    def get_window_sizes(self):
        return sorted(self.states.keys())

    def calc_states(self, tracklet_list, window_size, moving_thres=0):
        self.states[window_size] = []

        # Inital state
        t = tracklet_list[0]
        state = CarState(t.frame, t.cx, t.cy, (1, 0), 0, 0, False)
        self.states[window_size].append(state)

        # Intermediate states until window size is reached
        for i in range(1, window_size):
            prev_t = tracklet_list[i-1]
            t = tracklet_list[i]
            vec, norm = calc_vector((prev_t.x, prev_t.y), (t.x, t.y))
            is_moving = norm > moving_thres
            angle = angle_between(vec, (1, 0)) if is_moving else self.states[window_size][i-1].angle
            state = CarState(t.frame, t.cx, t.cy, (1, 0), 1, 0, int(False))
            self.states[window_size].append(state)

        # Regular calculation based on the window size
        for i in range(window_size, len(tracklet_list)):
            t = tracklet_list[i]
            comp_state = self.states[window_size][i-window_size]
            vec, norm = calc_vector((comp_state.cx, comp_state.cy), (t.cx, t.cy))
            is_moving = norm > moving_thres
            angle = angle_between(vec, (1,0)) if is_moving else self.states[window_size][i-1].angle
            state = CarState(t.frame, t.cx, t.cy, vec, norm, angle, int(is_moving))
            self.states[window_size].append(state)

    def get_state_list(self, window_size):
        return self.states[window_size]

    def get_attr_list(self, attribute, window_size):
        return [getattr(s, attribute) for s in self.states[window_size]]