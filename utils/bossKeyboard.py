import threading
from pynput import keyboard

class bossKeyboard:
    def __init__(self, press_key: list):
        '''初始化函数, 设置程序运行状态为True'''
        self.program_running = True
        self.press_key = press_key
        
    def on_press(self, key):
        """键盘按下的回调函数"""
        try:
            if key.char in self.press_key:
                print(f"\n检测到 {key.char} 键，主程序即将退出...")
                self.program_running = False
        except AttributeError:
            pass


    def on_release(self, key):
        '''摆设函数'''
        return


    def start_keyboard_listener(self):
        """启动键盘监听器"""
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()

    def startListen(self):
        '''
        This function is used to get the keyboard input from the user to control the boss.
        The function returns the key pressed by the user
        '''
        listener_thread = threading.Thread(target=self.start_keyboard_listener)
        listener_thread.daemon = True  # 设置为守护线程，主程序退出时自动结束
        listener_thread.start()
