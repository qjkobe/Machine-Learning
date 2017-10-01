# -*-coding:utf-8-*-
from pymouse import PyMouse
import pyHook
import pythoncom
import time
import thread
import msvcrt

flag = 0
m = PyMouse()
hm = pyHook.HookManager()


def onKeyBoardEvent(event):
    global flag
    print "MessageName:", event.MessageName
    print "Message:", event.Message
    print "Time:", event.Time
    print "Window:", event.Window
    print "WindowName:", event.WindowName
    print "Ascii:", event.Ascii, chr(event.Ascii)
    print "Key:", event.Key
    print "KeyID:", event.KeyID
    print "ScanCode:", event.ScanCode
    print "Extended:", event.Extended
    print "Injected:", event.Injected
    print "Alt", event.Alt
    print "Transition", event.Transition
    print "---"
    if event.Key == "S":
        click(10)

    return True


def click(num):
    for i in range(num):

        m.click(76, 41)
        time.sleep(5)
    return


def key_listen():
    hm.KeyDown = onKeyBoardEvent
    hm.HookKeyboard()
    pythoncom.PumpMessages()


if __name__ == "__main__":
    flag = 0
    # thread.start_new_thread(key_listen(), ("key_listen", 2))

    # thread.start_new_thread(click(10), ("click_event", 1))
    # click(10)
    # char = msvcrt.getch()
    # if char == "s":

    while 1:
        char = msvcrt.getch()
        if char == chr(27):
            break
        print char,
        if char == chr(13):
            print
    print 1


