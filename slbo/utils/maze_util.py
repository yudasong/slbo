def get_state_block(state):
    x = state[0].item() / 4
    y = state[1].item() / 4

    # if -1 < x < 1:
    #     x_block = 'low'
    # elif 1 < x < 3:
    #     x_block = 'mid'
    # elif 3 < x < 5:
    #     x_block = 'high'
    # else:
    #     raise Exception

    # if -1 < y < 1:
    #     y_block = 'left'
    # elif 1 < y < 3:
    #     y_block = 'center'
    # elif 3 < y < 5:
    #     y_block = 'right'
    # else:
    #     raise Exception

    # if x_block == 'low' and y_block == 'left':
    #     return 0
    # elif x_block == 'low' and y_block == 'center':
    #     return 1
    # elif x_block == 'low' and y_block == 'right':
    #     return 2
    # elif x_block == 'mid' and y_block == 'right':
    #     return 3
    # elif x_block == 'high' and y_block == 'right':
    #     return 4
    # elif x_block == 'high' and y_block == 'center':
    #     return 5
    # elif x_block == 'high' and y_block == 'left':
    #     return 6


    if -1 < x < 0:
        if -1 < y < 0:
            return 0
        elif 0 <= y < 1:
            return 1
        elif 1 <= y < 2:
            return 2
        elif 2 <= y < 3:
            return 3
        elif 3 <= y < 4:
            return 4
        elif 4 <= y < 5:
            return 5

    elif 0 <= x < 1:
        if -1 < y < 0:
            return 6
        elif 0 <= y < 1:
            return 7
        elif 1 <= y < 2:
            return 8
        elif 2 <= y < 3:
            return 9
        elif 3 <= y < 4:
            return 10
        elif 4 <= y < 5:
            return 11

    elif 1 <= x < 2:
        if 3 <= y < 4:
            return 12
        elif 4 <= y < 5:
            return 13

    elif 2 <= x < 3:
        if 3 <= y < 4:
            return 14
        elif 4 <= y < 5:
            return 15

    elif 3 <= x < 4:
        if -1 < y < 0:
            return 16
        elif 0 <= y < 1:
            return 17
        elif 1 <= y < 2:
            return 18
        elif 2 <= y < 3:
            return 19
        elif 3 <= y < 4:
            return 20
        elif 4 <= y < 5:
            return 21

    elif 4 <= x < 5:
        if -1 < y < 0:
            return 22
        elif 0 <= y < 1:
            return 23
        elif 1 <= y < 2:
            return 24
        elif 2 <= y < 3:
            return 25
        elif 3 <= y < 4:
            return 26
        elif 4 <= y < 5:
            return 27


def rate_buffer(states):
    visited_blocks = [get_state_block(state) for state in states]
    #print(visited_blocks)
    n_unique = len(set(visited_blocks))
    #print(n_unique)
    return n_unique / 7


class Coverage():
    def __init__(self):
        self.cov_list = []

    def rate_buffer(self,states):
        visited_blocks = [get_state_block(state) for state in states]
        #print(visited_blocks)
        self.cov_list = list(set(self.cov_list + visited_blocks))
        return len(self.cov_list) / 28



