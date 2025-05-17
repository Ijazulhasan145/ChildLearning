from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import default_storage
from django.http import HttpResponse
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch
import speech_recognition as sr
from gtts import gTTS
import os
from PIL import Image
import time

# Load BLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)


model_state_path = os.path.join(os.getcwd(), '4Epoch_best_vqa_model.pth')
if os.path.exists(model_state_path):
    model.load_state_dict(torch.load(model_state_path, map_location=torch.device('cpu')))  # Use 'cuda' if using GPU
    model.eval()  # Set the model to evaluation mode
else:
    raise FileNotFoundError("Trained model state file not found. Please ensure 'trained_model_state.pth' exists.")

# Ensure media directory exists
media_root = os.path.join(os.getcwd(), 'media')
if not os.path.exists(media_root):
    os.makedirs(media_root)


# Ensure media directory exists
#media_root = settings.MEDIA_ROOT
#os.makedirs(media_root, exist_ok=True)

def index(request):
    """Renders the main index page."""
    return render(request, 'childlearning_app/index.html')

def text_to_speech(answer):
    """Converts text to speech and returns an audio response."""
    try:
        tts = gTTS(text=answer, lang="en")
        audio_path = os.path.join(media_root, "answer.mp3")
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        print(f"gTTS Error: {e}")
        return None


def process_input(request):
    """Handles image and voice input, processes them, and returns response."""
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        if not image_file:
            return render(request, 'childlearning_app/index.html', {'error': 'No image uploaded'})

        # Ensure unique filename by appending a timestamp
        unique_filename = f"{int(time.time())}_{image_file.name}"
        image_path = default_storage.save(f"images/{unique_filename}", image_file)
        image_full_path = os.path.join(media_root, image_path)

        question = request.POST.get('question', '').strip()

        if not question and 'voice_question' in request.FILES:
            audio_file = request.FILES['voice_question']
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_file) as source:
                audio = recognizer.record(source)
            try:
                question = recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                question = "Could not understand the audio."
            except sr.RequestError:
                question = "Speech recognition service unavailable."

        if not question:
            return render(request, 'childlearning_app/index.html', {'error': 'No question provided'})

        try:
            image = Image.open(image_full_path).convert("RGB")
            inputs = processor(image, question, return_tensors="pt").to(device)
            out = model.generate(**inputs)
            answer = processor.decode(out[0], skip_special_tokens=True)

            voice_answer_path = text_to_speech(answer)
            voice_answer_url = "/media/answer.mp3" if voice_answer_path else None

            return render(request, 'childlearning_app/result.html', {
                'question': question,
                'answer': answer,
                'voice_answer_url': voice_answer_url,
                'image_url': f"/media/images/{unique_filename}"  # Ensure correct image display
            })
        except Exception as e:
            return render(request, 'childlearning_app/index.html', {'error': f'Error processing image: {str(e)}'})

    return render(request, 'childlearning_app/index.html')
