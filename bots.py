from pyautogui import *
import pyautogui
import time
import keyboard
import random
import win32api, win32con

#play around with it more tmr 


#pyautogui.displayMousePosition()


#p_1 X:  592 Y:  664 RGB: (197,  99, 117)
#p_2 X:  684 Y:  669 RGB: (195,  97, 117)
#P_3 X:  766 Y:  678 RGB: (165,  66,  71)
#P_4 X:  855 Y:  666 RGB: (200, 101, 117)




def click(x,y):
    win32api.SetCursorPos((x,y))
    #time.sleep(0.01)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
    time.sleep(0.01)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)


while keyboard.is_pressed('q') == False: # Press Q to stop 
    if pyautogui.pixel(592,650)[0] == 0:
        click(592,650)
    if pyautogui.pixel(684,650)[0] == 0:
        click(684,650)
    if pyautogui.pixel(766,650)[0] == 0:
        click(766,650)
    if pyautogui.pixel(855,650)[0] == 0:
        click(855,650)

        




