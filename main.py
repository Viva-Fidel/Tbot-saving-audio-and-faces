import cv2
import os
import librosa
import logging
import requests
import urllib

import mediapipe as mp
import numpy as np
import soundfile as sf

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, CallbackContext, filters

'''
Бот для telegram. Сохраняет аудиосообщения из диалогов в базу данных на диск по идентификаторам пользователей.
Конвертирует все аудиосообщения в формат wav с частотой дискретизации 16kHz
Определяет есть ли лицо на отправляемых фотографиях или нет, сохраняет только те, где оно есть
Для распознавания лиц используется библиотека Mediapipe

В строке ApplicationBuilder().token('').build() вместо кавычек надо указать ключ от своего бота
'''

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Вызывает команду /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id,
                                   text="Привет, отправь аудисообщение или фото с лицом, чтобы я мог их сохранить")

# Вызывает команду /help
async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id,
                                   text="Нужно всего лишь отправить аудиосообщение, либо селфи с собой в диалог")

# Получает аудиофайл, сохраняет и преобразовывает его
async def get_voice(update: Update, context: CallbackContext):
    path = f"{update.message.from_user.id}"
    check_for_path = os.path.exists(path) # Проверяем есть ли папка c названием, как id пользователя в telegram
    if not check_for_path:
        os.makedirs(path)
    counter = len(os.listdir(path))

    audio_file = await context.bot.get_file(update.message.voice.file_id)

    file_name = f'{path}/audio_message_{counter}.wav' # Сохраняем в wav
    await audio_file.download_to_drive(file_name)
    data, samplerate = librosa.load(file_name)
    sf.write(file_name, data, 16000) # Преобразовываем файл, чтобы он имел частоту дискретизации 16kHz
    await update.message.reply_text('Voice message saved')


async def get_photo(update: Update, context: CallbackContext):
    photo = await context.bot.get_file(update.message.photo[-1].file_id)
    get_photo = urllib.request.urlopen(photo.file_path) # Открываем фото и преобразовываем в массив
    make_array_from_photo = np.asarray(bytearray(get_photo.read()), dtype=np.uint8)
    img = cv2.imdecode(make_array_from_photo, -1)
    check_for_face = await detect_face(img) # Проверяем есть ли на фото лицо
    if check_for_face:
        cv2.imwrite(f'{photo.file_unique_id}.jpg', img) # Если лицо обнаружено, сохраняем файл
        await update.message.reply_text('Photo saved')


async def detect_face(img):
    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.7) as face_detection:
        results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # С помощью библиотеки Media Pipe проверяем наличии лица на фото
        if not results.detections:
            print('No face')
            return False
        else:
            print('Face detected')
            return True


if __name__ == '__main__':
    application = ApplicationBuilder().token('').build() # Вставьте токен для вашего бота

    start_handler = CommandHandler('start', start)
    help_handler = CommandHandler('help', help)
    application.add_handler(start_handler)
    application.add_handler(help_handler)

    application.add_handler(MessageHandler(filters.VOICE, get_voice))
    application.add_handler(MessageHandler(filters.PHOTO, get_photo))

    application.run_polling()
