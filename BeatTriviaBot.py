"""
Bot to transcribe audio messages using Hugging Face's whisper-large-v3 model.

```python
python BeatTriviaBot.py
```
```

Press Ctrl-C on the command line to stop the bot.
"""

import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
import requests
from telegram.ext import Application, ContextTypes, MessageHandler, filters, CommandHandler, CallbackQueryHandler
from keys import TELEGRAM_KEY, HUGGING_FACE_KEY, OPENAI_KEY
from pprint import pprint
from openai import OpenAI
import torch


MAX_CHAT_HISTORY = 3
VERBOSE = True
LOCAL_ASR = False
CUDA_AVAILABLE = torch.cuda.is_available()


global USER_MESSAGES    # user message history

USER_MESSAGES = dict()    # dict of chat/group IDs and their messages
USER_LAST_QUEST = dict() # dict of user id with the last question
USER_PROGRESSION = dict() # dict of user id with the last step # 1 is question, 2 is MCQ, 3 is anwer
USER_DIFFICULTY = dict() 

headers = {"Authorization": f"Bearer {HUGGING_FACE_KEY}"}

# prepare LLM
client = OpenAI(api_key=OPENAI_KEY)

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# prepare ASR
if LOCAL_ASR:
    from transformers import pipeline
    import audiofile
    import librosa

    if CUDA_AVAILABLE:
        asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=0)
    else:
        asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
    asr_rate = 16000

# querying ASR
def query_asr(filename):

    if LOCAL_ASR:

        signal, sampling_rate = audiofile.read(filename)
        if sampling_rate != asr_rate:
            signal = librosa.resample(signal, orig_sr=sampling_rate, target_sr=asr_rate)
        output = asr_pipe(signal, generate_kwargs={"language": "english"})
        return output

    else:
        # Hugging Face endpoint
        with open(filename, "rb") as f:
            data = f.read()
        response = requests.post(
            "https://api-inference.huggingface.co/models/openai/whisper-large-v3", 
            headers=headers, 
            data=data
        )
        return response.json()

def prompt_input_text(input_text, user_id):

    # USER_LAST_QUEST = dict() # dict of user id with the last question
    # USER_PROGRESSION = dict() # dict of user id with the last step # 1 is question, 2 is MCQ, 3 is anwer
    # do a dic about difficulty level of question
    if user_id not in USER_DIFFICULTY:
        USER_DIFFICULTY[user_id] = 2

    if user_id in USER_LAST_QUEST:
        last_quest = USER_LAST_QUEST[user_id]
    else: 
        USER_LAST_QUEST[user_id] = ""
        USER_PROGRESSION[user_id] = 3

    if USER_PROGRESSION[user_id] == 1: # question
        # input_text = f"Is '{input_text}' the correct answer to the question '{last_quest}'? If yes, congrats the user with a joke related to the question. If no, give him a MCQ with 4 shuffled choices (3 false and 1 true). Don't give the answer."
        input_text = f"Based on the previously asked question '{last_quest}', create a multiple-choice question with four options: one correct answer and three distractors. Ensure the options are relevant and plausible but clearly distinguishable from the correct answer. Shuffle the options to randomize their order. Do not reveal which option is correct in the response."
        USER_PROGRESSION[user_id] = 2

    elif USER_PROGRESSION[user_id] == 2: # MCQ
        input_text = f"Is '{input_text}' the correct answer to the question '{last_quest}'? If yes, congratulate the user with a interesting fact related to the theme of the question. If no, provide the correct answer and a brief explanation."
        USER_PROGRESSION[user_id] = 3

    elif USER_PROGRESSION[user_id] == 3: # answer should be given
        input_text += "Based on the user's input, generate a cultural knowledge question. \
            If the user specifies a theme from the following: sports, politics, music, art, history, geography, or science. \
            If no theme is specified, randomly select one of these themes and create a question. \
            Develop a question that delves into that theme, exploring facets such as historical milestones, cultural impacts, pivotal figures, or notable events. \
            To enhance learning and engagement, vary the type of question: it could be a 'true or false', a specific date, a 'yes or no', a number, or a name of a historical figure or an event. "

        USER_PROGRESSION[user_id] = 1
    
    return input_text
        

# querying LLM
def query_llm(input_text, user_id):
    
    prompted_text = prompt_input_text(input_text, user_id)

    # add to message history
    if user_id in USER_MESSAGES:
        USER_MESSAGES[user_id].append({"role": "user", "content": prompted_text})
    else:
        USER_MESSAGES[user_id] = [{"role": "user", "content": prompted_text}]

    pprint(USER_MESSAGES[user_id])

    # prompt LLM
    # -- OpenAI
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=USER_MESSAGES[user_id]
    )
    text_response = response.choices[0].message.content
    
    # update message history
    USER_MESSAGES[user_id].append({"role": "assistant", "content": text_response})
    if USER_PROGRESSION[user_id] == 1:
        USER_LAST_QUEST[user_id] = text_response

    if VERBOSE:
        pprint(USER_MESSAGES[user_id])

    # clear chat history (if too long)
    if len(USER_MESSAGES[user_id]) > 2 * MAX_CHAT_HISTORY:
        # remove bot response
        USER_MESSAGES[user_id].pop(0)
        # remove question
        USER_MESSAGES[user_id].pop(0)

    return text_response


async def voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """STEP 1) Speech to text."""
    audio_file = await update.message.voice.get_file()

    # load audio into numpy array
    tmp_file = "voice_note.ogg"
    await audio_file.download_to_drive(tmp_file)

    # transcription
    output = query_asr(tmp_file)
    try:
        output = output["text"]
    except:
        output = "Sorry, I could not understand the audio message. Please try again."
        await update.message.reply_text(output)
        return

    """STEP 2) Prompt LLM."""
    text_response = query_llm(output, update.message.from_user.id)

    # respond text through Telegram
    await update.message.reply_text(text_response)


async def text_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # prompt LLM
    input_text = update.message.text
    user_id = update.message.from_user.id
    text_response = query_llm(input_text, user_id)

    # respond text through Telegram
    await update.message.reply_text(text_response)


def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TELEGRAM_KEY).build()

    # voice input
    application.add_handler(
        MessageHandler(filters.VOICE & ~filters.COMMAND, voice, block=True)
    )

    # text input
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_input, block=True))


    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()