
red           = "\033[0;31;40m"
green         = "\033[0;32;40m"
yellow        = "\033[0;33;40m"
blue          = "\033[0;34;40m"
magenta       = "\033[0;35;40m"
cyan          = "\033[0;36;40m"
lgray         = "\033[0;37;40m"
gray          = "\033[0;37;90m"
back_gray     = "\033[0;90;47m"
lred          = "\033[0;91;40m"
lgreen        = "\033[0;92;40m"
lyellow       = "\033[0;93;40m"
lblue         = "\033[0;94;40m"
lmagenta      = "\033[0;95;40m"
lcyan         = "\033[0;96;40m"
back_red      = "\033[0;37;41m"
back_green    = "\033[0;37;42m"
back_yellow   = "\033[0;37;43m"
back_blue     = "\033[0;37;44m"
back_magenta  = "\033[0;37;45m"
back_cyan     = "\033[0;37;46m"
back_lgray    = "\033[0;37;47m"
back_lred     = "\033[0;37;91m"
back_lgreen   = "\033[0;37;92m"
back_lyellow  = "\033[0;37;93m"
back_lblue    = "\033[0;37;94m"
back_lmagenta = "\033[0;37;95m"
back_lcyan    = "\033[0;37;96m"

back_llgray   = "\033[7;35;40m"

reset = "\033[0m"

bold = "\033[1m"
nonbold = "\033[0m"


def cprint(*content, background=False, color='white', sep=' ', end='\n'):
    back = 40
    fore = 37
    if color == 'red':
        fore = 31
    elif color == 'green':
        fore = 32
    elif color == 'yellow':
        fore = 33
    elif color == 'blue':
        fore = 34
    elif color == 'magenta':
        fore = 35
    elif color == 'cyan':
        fore = 36
    elif color == 'lgray':
        fore = 37
    elif color == 'gray':
        fore = 90
    if color == 'lred':
        fore = 91
    elif color == 'lgreen':
        fore = 92
    elif color == 'lyellow':
        fore = 93
    elif color == 'lblue':
        fore = 94
    elif color == 'lmagenta':
        fore = 95
    elif color == 'lcyan':
        fore = 96
    if background:
        back = fore + 10
        fore = 37
    print('\033[0;', fore, ';', back, 'm', sep='', end='')
    print(*content, sep=sep, end='')
    print('\033[0m', sep='', end=end)
        