

from ConvertSpeechToSignal import ConvertSpeechToSignal
import serial
import time

if __name__ == '__main__':
    DataSerial = serial.Serial('/dev/cu.HC-06', 9600)
    convert = ConvertSpeechToSignal()
    result = convert.convert()
    print("Tín hiệu truyền cho robot là: ", result)
    for i in result:
        DataSerial.write(i.encode())
