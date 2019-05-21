import socket
import process
import time
import visdom
import sys
import numpy as np

vis = None
port = None


def port_used(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    if result == 0:
        sock.close()
        return True
    else:
        return False


def alloc_port(start_from=7000):
    while True:
        if port_used(start_from):
            print("Port already used: %d" % start_from)
            start_from += 1
        else:
            return start_from


def wait_for_port(port, timeout=5):
    star_time = time.time()
    while not port_used(port):
        if time.time() - star_time > timeout:
            return False

        time.sleep(0.1)
    return True


def start(on_port=None):
    global vis
    global port
    assert vis is None, "Cannot start more than 1 visdom servers."

    port = alloc_port() if on_port is None else on_port

    print("Starting Visdom server on %d" % port)
    process.run("%s -m visdom.server -p %d" % (sys.executable, port))
    if not wait_for_port(port):
        print("ERROR: failed to start Visdom server. Server not responding.")
        return
    print("Done.")

    vis = visdom.Visdom(port=port)


def _start_if_not_running():
    if vis is None:
        start()


class Plot2D:
    TO_SAVE = ["x", "y", "curr_accu", "curr_cnt", "legend"]

    def __init__(self, name, store_interval=1, legend=None, xlabel=None, ylabel=None):
        _start_if_not_running()

        self.x = []
        self.y = []
        self.store_interval = store_interval
        self.curr_accu = None
        self.curr_cnt = 0
        self.name = name
        self.legend = legend
        self.xlabel = xlabel
        self.ylabel = ylabel

        self.replot = False
        self.visplot = None
        self.last_vis_update_pos = 0

    def set_legend(self, legend):
        if self.legend != legend:
            self.legend = legend
            self.replot = True

    def _send_update(self):
        if not self.x or vis is None:
            return

        if self.visplot is None or self.replot:
            opts = {
                "title": self.name
            }
            if self.xlabel:
                opts["xlabel"] = self.xlabel
            if self.ylabel:
                opts["ylabel"] = self.ylabel

            if self.legend is not None:
                opts["legend"] = self.legend
            if vis is not None:
                self.visplot = vis.line(X=np.asfarray(self.x), Y=np.asfarray(self.y), opts=opts, win=self.visplot)
            self.replot = False
        else:
            if vis is not None:
                vis.line(
                    X=np.asfarray(self.x[self.last_vis_update_pos:]),
                    Y=np.asfarray(self.y[self.last_vis_update_pos:]),
                    win=self.visplot,
                    update='append'
                )

        self.last_vis_update_pos = len(self.x) - 1

    def add_point(self, x, y):
        if not isinstance(y, list):
            y = [y]

        if self.curr_accu is None:
            self.curr_accu = [0.0] * len(y)

        if len(self.curr_accu) < len(y):
            # The number of curves increased.
            need_to_add = (len(y) - len(self.curr_accu))

            self.curr_accu += [0.0] * need_to_add
            count = len(self.x)
            if count>0:
                self.replot = True
                if not isinstance(self.x[0], list):
                    self.x = [[x] for x in self.x]
                    self.y = [[y] for y in self.y]

                nan = float("nan")
                for a in self.x:
                    a += [nan] * need_to_add
                for a in self.y:
                    a += [nan] * need_to_add
        elif len(self.curr_accu) > len(y):
            y = y[:] + [float("nan")] * (len(self.curr_accu) - len(y))

        self.curr_accu = [self.curr_accu[i] + y[i] for i in range(len(y))]
        self.curr_cnt += 1
        if self.curr_cnt == self.store_interval:
            if len(y) > 1:
                self.x.append([x] * len(y))
                self.y.append([a / self.curr_cnt for a in self.curr_accu])
            else:
                self.x.append(x)
                self.y.append(self.curr_accu[0] / self.curr_cnt)

            self.curr_accu = [0.0] * len(y)
            self.curr_cnt = 0

            self._send_update()

    def state_dict(self):
        s = {k: self.__dict__[k] for k in self.TO_SAVE}
        return s

    def load_state_dict(self, state):
        if state is None:
            return
        if self.legend is not None:
            # Load legend only if not given in the constructor.
            state.legend = self.legend
        self.__dict__.update(state)
        self.last_vis_update_pos = 0

        # Read old format
        if not isinstance(self.curr_accu, list) and self.curr_accu is not None:
            self.curr_accu = [self.curr_accu]

        self._send_update()


class Text:
    def __init__(self, title):
        _start_if_not_running()

        self.win = None
        self.title = title
        self.curr_text = ""

    def set(self, text):
        if vis is None:
            return

        self.curr_text = text
        if vis is not None:
            if self.win is None:
                self.win = vis.text(text, opts=dict(
                    title=self.title
                ))
            else:
                vis.text(text, win=self.win)

    def state_dict(self):
        return {"text": self.curr_text}

    def load_state_dict(self, state):
        if state is None:
            return
        self.set(state["text"])

    def __call__(self, text):
        self.set(text)
