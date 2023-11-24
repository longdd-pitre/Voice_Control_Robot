import speech_recognition as sr
from unidecode import unidecode
from CustomTokenizer import CustomTokenizer
import serial
import time
import speech_recognition as sr
from unidecode import unidecode

class ConvertSpeechToSignal:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def convert(self):
        with sr.Microphone() as source:
            print("Lenh dieu khien...")
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)

            try:
                text_with_diacritics = self.recognizer.recognize_google(audio, language="vi-VN")
                text_without_diacritics = unidecode(text_with_diacritics).lower()
                print("Text: {}".format(text_without_diacritics))
                tokenizer = CustomTokenizer()
                tokens = tokenizer.tokenize(text_without_diacritics)
                result = ' '.join(tokens)
                return result
            except sr.UnknownValueError:
                print("Không nhận diện được giọng nói")
                return None


convert = ConvertSpeechToSignal()
result = convert.convert()
print("Tín hiệu truyền cho robot là: ", result)

