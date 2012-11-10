class Color(object):
    BLUE = '\033[95m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    
    @staticmethod
    def blue(input):
        return Color.BLUE + input + Color.ENDC

    @staticmethod
    def green(input):
        return Color.GREEN + input + Color.ENDC

    @staticmethod
    def yellow(input):
        return Color.YELLOW + input + Color.ENDC

    @staticmethod
    def red(input):
        return Color.RED + input + Color.ENDC

class Output(object):
    import curses

    def __init__(self, xpos, ypos, height, width):
        self.pos = (xpos, ypos)
        self.dim = (height, width)
        
        self.stdscr = Output.curses.initscr()

        Output.curses.noecho()
        Output.curses.cbreak()

        self.stdscr.keypad(1)

        self.stdscr.nodelay(1)

    def __del__(self):
        Output.curses.nocbreak()
        self.stdscr.keypad(0)
        Output.curses.echo()
        Output.curses.endwin()